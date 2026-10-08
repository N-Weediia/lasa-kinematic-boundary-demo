"""Run a deliberately favorable synthetic example; writes no private data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kinematic_boundary import boundary_metrics, local_peaks, refine_boundaries


def make_synthetic_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct sharp, known kinematic transitions and deliberately shifted peaks."""
    rng = np.random.default_rng(2026)
    lengths = np.array([50, 50, 50, 50, 50])
    truth = np.r_[0, np.cumsum(lengths)[:-1]]
    phase = np.repeat(np.arange(len(lengths)), lengths)
    n = phase.size
    speed = np.array([0.02, 0.25, 0.08, 0.30, 0.03])[phase]
    turn = np.array([0.0, 0.0, 0.05, 0.9, 0.0])[phase]
    speed += rng.normal(0.0, 0.001, n)
    turn += rng.normal(0.0, 0.001, n)
    model = np.full(n, 0.02)
    for boundary, offset in zip(truth, [0, 4, -4, 4, -4]):
        center = int(np.clip(boundary + offset, 0, n - 1))
        model += 0.9 * np.exp(-0.5 * ((np.arange(n) - center) / 1.5) ** 2)
    # A low-confidence false candidate illustrates that kinematics do not
    # create additional candidates; it remains an explicit learned proposal.
    model += 0.45 * np.exp(-0.5 * ((np.arange(n) - 20) / 1.5) ** 2)
    return truth, speed, turn, np.clip(model, 0.0, 1.0)


def make_wrong_cue_counterexample() -> dict:
    """Show that overestimated QC plus a deliberately wrong cue can hurt."""
    n = 80
    t = np.arange(n)
    truth = [0, 44]
    model = np.full(n, 0.01)
    model[0], model[44] = 1.0, 0.95
    speed = 1.0 / (1.0 + np.exp(-(t - 38.0)))
    turn = np.zeros(n)
    turn[38] = 20.0  # Intentionally inconsistent with the truth boundary.
    baseline = local_peaks(model, threshold=0.7)
    _, wrong_cue, _ = refine_boundaries(
        model, speed, turn, quality=1.0, threshold=0.7,
        snap_radius=10, min_eventness=0.4, smooth_width=1,
    )
    return {
        "warning": "Synthetic failure case: QC is intentionally overestimated and the motion cue is wrong.",
        "truth_boundaries": truth,
        "baseline_target_boundary_error_frames": abs(baseline[1] - truth[1]),
        "wrong_cue_target_boundary_error_frames": abs(wrong_cue[1] - truth[1]),
        "baseline": boundary_metrics(baseline, truth, tolerance=5),
        "wrong_cue_assisted": boundary_metrics(wrong_cue, truth, tolerance=5),
        "candidate_count_baseline": len(baseline),
        "candidate_count_wrong_cue": len(wrong_cue),
    }


def main() -> None:
    truth, speed, turn, model = make_synthetic_case()
    baseline = local_peaks(model, threshold=0.7)
    _, guided, eventness = refine_boundaries(
        model, speed, turn, quality=1.0, threshold=0.7, snap_radius=6
    )
    _, low_quality, _ = refine_boundaries(
        model, speed, turn, quality=0.2, threshold=0.7, snap_radius=6
    )
    result = {
        "case": "constructed_synthetic_mechanism_example",
        "warning": "The signal is deliberately favorable and is not scientific evidence or a LaSA benchmark.",
        "threshold": 0.7,
        "truth_boundaries": truth.tolist(),
        "baseline": boundary_metrics(baseline, truth, tolerance=5),
        "kinematic_assisted": boundary_metrics(guided, truth, tolerance=5),
        "low_quality_fallback_exact": low_quality == baseline,
        "kinematic_eventness_max": float(np.max(eventness)),
        "overestimated_qc_wrong_cue_counterexample": make_wrong_cue_counterexample(),
        "candidate_count_baseline": len(baseline),
        "candidate_count_assisted": len(guided),
        "note": "LaSA peaks remain the candidate set; kinematics can only adjust their locations.",
    }
    out = Path(__file__).resolve().parent / "demo_output.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
