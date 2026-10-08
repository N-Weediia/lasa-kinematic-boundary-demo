"""Apply and exercise the patch against a clean, hash-verified LaSA checkout."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = Path(__file__).resolve().with_name("integration.patch")
EXPECTED_GIT_BLOBS = {
    "libs/postprocess.py": "2405b34e6e84c90fc2dd093b67f1083a2f2c7196",
    "libs/metric.py": "fa591a04afa9278fc0988aec7980619c65a93160",
    "libs/transformer.py": "bff4689c01cab33943074d6b0696f5000b234388",
}


def git_blob_sha(data: bytes) -> str:
    # GitHub stores LF in the commit; normalize ordinary Windows CRLF checkout.
    data = data.replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def patched_sources(upstream: Path) -> dict[str, str]:
    sources = {name: (upstream / name).read_text(encoding="utf-8") for name in EXPECTED_GIT_BLOBS}
    for name, source in sources.items():
        digest = git_blob_sha((upstream / name).read_bytes())
        if digest != EXPECTED_GIT_BLOBS[name]:
            raise RuntimeError(f"{name} is not the verified clean LaSA source (blob {digest})")

    post = sources["libs/postprocess.py"]
    post = post.replace(
        "import torch\n\nfrom .metric import argrelmax",
        "import torch\n\nfrom kinematic_boundary import refine_boundaries\n\nfrom .metric import argrelmax",
        1,
    )
    post = post.replace(
        "        masks: np.ndarray,\n    ) -> np.ndarray:",
        "        masks: np.ndarray,\n        com_speed: np.ndarray = None,\n"
        "        angular_velocity: np.ndarray = None,\n        quality: np.ndarray = None,\n"
        "    ) -> np.ndarray:",
        1,
    )
    old_block = (
        "        for i, (output, pred, boundary, mask) in enumerate(\n"
        "            zip(outputs, preds, boundaries, masks)\n"
        "        ):\n"
        "            boundary = boundary[mask]\n"
        "            idx = argrelmax(boundary, threshold=self.boundary_th)\n"
    )
    new_block = (
        "        for i, (output, pred, boundary, mask) in enumerate(\n"
        "            zip(outputs, preds, boundaries, masks)\n"
        "        ):\n"
        "            mask_array = np.asarray(mask, dtype=bool)\n"
        "            valid_mask = mask_array.reshape(-1)\n"
        "            # Preserve upstream's (1,T) boolean indexing and also accept (T,).\n"
        "            if boundary.shape == mask_array.shape:\n"
        "                boundary_valid = boundary[mask_array].reshape(-1)\n"
        "            else:\n"
        "                boundary_valid = boundary.reshape(-1)[valid_mask]\n\n"
        "            cue_ok = (\n"
        "                com_speed is not None\n"
        "                and angular_velocity is not None\n"
        "                and quality is not None\n"
        "            )\n"
        "            if cue_ok:\n"
        "                speed_batch = np.asarray(com_speed)\n"
        "                turn_batch = np.asarray(angular_velocity)\n"
        "                q_batch = np.asarray(quality)\n"
        "                cue_ok = (\n"
        "                    speed_batch.ndim == 2\n"
        "                    and turn_batch.shape == speed_batch.shape\n"
        "                    and speed_batch.shape[0] == len(preds)\n"
        "                    and speed_batch.shape[1] == pred.shape[0]\n"
        "                    and (q_batch.ndim == 0 or q_batch.size == len(preds))\n"
        "                )\n"
        "            if cue_ok:\n"
        "                q = float(q_batch if q_batch.ndim == 0 else q_batch.reshape(-1)[i])\n"
        "                _, idx, _ = refine_boundaries(\n"
        "                    boundary_valid,\n"
        "                    speed_batch[i][valid_mask],\n"
        "                    turn_batch[i][valid_mask],\n"
        "                    quality=q,\n"
        "                    threshold=self.boundary_th,\n"
        "                )\n"
        "            else:\n"
        "                idx = argrelmax(boundary_valid.copy(), threshold=self.boundary_th)\n"
    )
    if old_block not in post:
        raise RuntimeError("could not locate the clean upstream refinement block")
    post = post.replace(old_block, new_block, 1)

    metric = sources["libs/metric.py"]
    metric = metric.replace("dtype=np.bool)", "dtype=np.bool_)")
    if metric == sources["libs/metric.py"]:
        raise RuntimeError("expected deprecated np.bool dtype was not found")
    return {"libs/postprocess.py": post, "libs/metric.py": metric}


def generate_patch(upstream: Path) -> str:
    modified = patched_sources(upstream)
    chunks = []
    for name in ("libs/metric.py", "libs/postprocess.py"):
        before = (upstream / name).read_text(encoding="utf-8").splitlines(keepends=True)
        after = modified[name].splitlines(keepends=True)
        chunks.append(f"diff --git a/{name} b/{name}\n")
        chunks.extend(difflib.unified_diff(before, after, fromfile=f"a/{name}", tofile=f"b/{name}"))
    return "".join(chunks)


def exercise(upstream: Path) -> None:
    expected = generate_patch(upstream)
    if not PATCH.exists() or PATCH.read_text(encoding="utf-8") != expected:
        raise RuntimeError("integration.patch does not match this clean upstream checkout")

    sandbox = ROOT / "integration" / (".lasa-integration-" + uuid.uuid4().hex)
    sandbox.mkdir()
    try:
        libs = sandbox / "libs"
        libs.mkdir()
        for name in EXPECTED_GIT_BLOBS:
            canonical = (upstream / name).read_bytes().replace(b"\r\n", b"\n")
            (libs / Path(name).name).write_bytes(canonical)
        (libs / "__init__.py").write_text("", encoding="utf-8")
        shutil.copy2(ROOT / "kinematic_boundary.py", sandbox / "kinematic_boundary.py")
        subprocess.run(["git", "apply", str(PATCH)], cwd=sandbox, check=True, capture_output=True, text=True)
        inserted = False
        try:
            sys.path.insert(0, str(sandbox))
            inserted = True
            import numpy as np
            from libs.postprocess import PostProcessor

            n, classes, frames = 2, 2, 48
            outputs = np.zeros((n, classes, frames), dtype=np.float32)
            outputs[:, 0, :24] = 1.0
            outputs[:, 1, 24:] = 1.0
            boundaries = np.zeros((n, 1, frames), dtype=np.float32)
            boundaries[:, 0, 0] = 1.0
            boundaries[:, 0, 26] = 0.9
            masks = np.zeros((n, 1, frames), dtype=bool)
            masks[0, 0, :] = True
            masks[1, 0, :35] = True
            post = PostProcessor("refinement_with_boundary", boundary_th=0.7)
            baseline = post(outputs, boundaries=boundaries, masks=masks)
            fallback = post(
                outputs, boundaries=boundaries, masks=masks,
                com_speed=np.zeros((n, frames)),
                angular_velocity=np.zeros((n, frames)),
                quality=np.array([0.1, 0.2]),
            )
            speed = np.zeros((n, frames), dtype=float)
            speed[:, 24:] = 1.0
            angular = np.zeros((n, frames), dtype=float)
            angular[:, 24] = 10.0
            active = post(
                outputs, boundaries=boundaries, masks=masks,
                com_speed=speed,
                angular_velocity=angular,
                quality=np.array([0.9, 0.9]),
            )
            if baseline.shape != (n, frames) or not np.array_equal(baseline, fallback):
                raise AssertionError("baseline/fallback mismatch through actual LaSA PostProcessor")
            if active.shape != (n, frames):
                raise AssertionError("active hook returned an unexpected prediction shape")
            if np.array_equal(active[0], baseline[0]):
                raise AssertionError("active cue did not change the shifted-candidate refinement")
            print("PASS: patched clean LaSA PostProcessor imports and runs with (N,1,T) masks and (N,T) cues")
            print("PASS: missing-cue baseline and low-QC fallback produce identical class predictions")
            print("PASS: active kinematic event changes majority-vote refinement for a shifted candidate")
            print("PASS: batched quality vector and shorter valid sequence are handled")
        finally:
            if inserted:
                sys.path.remove(str(sandbox))
    finally:
        shutil.rmtree(sandbox)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True, type=Path,
                        help="clean LaSA checkout at the verified upstream commit")
    parser.add_argument("--write-patch", action="store_true",
                        help="regenerate the checked-in integration.patch from verified source")
    args = parser.parse_args()
    patch = generate_patch(args.upstream_dir)
    if args.write_patch:
        with PATCH.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(patch)
    exercise(args.upstream_dir)


if __name__ == "__main__":
    main()
