"""Conservative, NumPy-only kinematic assistance for LaSA boundary peaks.

LaSA's learned boundary peaks remain the candidate set. This module may move
an existing candidate to a nearby kinematic event when reconstruction QC is
reliable; it never creates a boundary from kinematics alone.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (T,)")
    return array


def local_peaks(probability: Sequence[float], threshold: float = 0.7) -> list[int]:
    """Match LaSA's strict local-max rule, including frame zero and threshold."""
    p = _vector(probability, "probability").copy()
    if p.size == 0:
        return []
    p[~np.isfinite(p)] = 0.0
    p[p < threshold] = 0.0
    if p.size == 1:
        return [0]
    peaks = [0]
    for index in range(1, p.size - 1):
        if p[index - 1] < p[index] and p[index + 1] < p[index]:
            peaks.append(index)
    return peaks


def _interpolate_finite(values: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return None
    positions = np.arange(values.size)
    return np.interp(positions, positions[valid], values[valid])


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width == 1:
        return values.copy()
    kernel = np.ones(width, dtype=float) / width
    left = width // 2
    right = width - 1 - left
    return np.convolve(np.pad(values, (left, right), mode="edge"), kernel, mode="valid")


def _scale_robust(values: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(values, [5.0, 95.0])
    # Sparse transition events can occupy fewer than five percent of frames;
    # if the upper percentile collapses onto the floor, scale by the observed
    # maximum instead. Constant signals still map to zero.
    if hi <= lo + 1e-12:
        hi = float(np.max(values))
    if not np.isfinite(lo) or hi <= lo + 1e-12:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def kinematic_eventness(
    com_speed: Sequence[float],
    angular_velocity: Sequence[float],
    *,
    smooth_width: int = 3,
) -> np.ndarray:
    """Compute bounded eventness from speed change and turn angular speed.

    Missing values are interpolated only when at least three finite samples
    remain in each signal. Otherwise a zero cue is returned, leaving learned
    boundary candidates unchanged in :func:`refine_boundaries`.
    """
    speed = _vector(com_speed, "com_speed")
    turn = _vector(angular_velocity, "angular_velocity")
    if speed.size != turn.size:
        raise ValueError("com_speed and angular_velocity must have equal length")
    if speed.size < 3:
        raise ValueError("kinematic signals require at least 3 samples")
    speed = _interpolate_finite(speed)
    turn = _interpolate_finite(turn)
    if speed is None or turn is None:
        return np.zeros(len(com_speed), dtype=float)
    speed = _smooth(speed, smooth_width)
    turn = _smooth(turn, smooth_width)
    eventness = 0.55 * _scale_robust(np.abs(np.gradient(speed)))
    eventness += 0.45 * _scale_robust(np.abs(turn))
    return np.clip(_smooth(eventness, smooth_width), 0.0, 1.0)


def refine_boundaries(
    model_probability: Sequence[float],
    com_speed: Sequence[float] | None = None,
    angular_velocity: Sequence[float] | None = None,
    quality: float | None = None,
    *,
    threshold: float = 0.7,
    snap_radius: int = 6,
    min_eventness: float = 0.65,
    smooth_width: int = 3,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Return fused diagnostic scores, conservative peaks, and eventness.

    QC is externally supplied reconstruction confidence in [0, 1]. Missing
    cues or QC, and QC below 0.5, return the exact LaSA-style candidate peaks.
    In the active path, only existing learned peaks can move, and only to a
    nearby strict eventness maximum. No kinematic-only boundary is introduced.
    """
    p = _vector(model_probability, "model_probability")
    p = np.clip(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    baseline = local_peaks(p, threshold)
    zero_event = np.zeros(p.size, dtype=float)
    if com_speed is None or angular_velocity is None or quality is None:
        return p.copy(), baseline, zero_event
    q = float(quality)
    if not np.isfinite(q) or q < 0.5 or q > 1.0:
        return p.copy(), baseline, zero_event

    speed = _vector(com_speed, "com_speed")
    turn = _vector(angular_velocity, "angular_velocity")
    if speed.size != p.size or turn.size != p.size:
        raise ValueError("all signals must have the same number of frames")
    if p.size < 3:
        return p.copy(), baseline, zero_event
    eventness = kinematic_eventness(speed, turn, smooth_width=smooth_width)

    # The fused trace is provided for inspection only. Candidate creation and
    # selection remain tied to LaSA's peaks to avoid cue-driven over-segmentation.
    alpha = 0.25 * q
    fused = np.clip((1.0 - alpha) * p + alpha * eventness, 0.0, 1.0)
    # Frame zero is an implementation convention in LaSA, not a measured
    # kinematic event and therefore must never attract a later candidate.
    event_peaks = [peak for peak in local_peaks(eventness, min_eventness) if peak > 0]
    refined = [0] if baseline else []
    for peak in baseline:
        if peak == 0:
            continue
        nearby = [candidate for candidate in event_peaks
                  if abs(candidate - peak) <= int(snap_radius)]
        if nearby:
            best = max(nearby, key=lambda candidate: eventness[candidate])
            # Move only when the event peak has clear evidence and is better
            # supported than the current location. Otherwise retain baseline.
            if eventness[best] >= min_eventness and eventness[best] > eventness[peak]:
                peak = best
        refined.append(int(peak))
    return fused, sorted(set(refined)), eventness


def boundary_metrics(predicted: Iterable[int], truth: Iterable[int], tolerance: int = 5) -> dict:
    """One-to-one matching metrics for an explicitly synthetic illustration."""
    pred = sorted(set(int(x) for x in predicted))
    gt = sorted(set(int(x) for x in truth))
    used: set[int] = set()
    errors: list[int] = []
    for candidate in pred:
        options = [(abs(candidate - target), i) for i, target in enumerate(gt) if i not in used]
        if options:
            error, index = min(options)
            if error <= tolerance:
                used.add(index)
                errors.append(error)
    tp = len(errors)
    fp, fn = len(pred) - tp, len(gt) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"predicted": pred, "truth": gt, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_matched_error_frames": float(np.mean(errors)) if errors else None}
