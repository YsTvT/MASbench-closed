"""Shared benchmark metrics for Minecraft, Bench2 and Bench3."""

from statistics import mean, median
import math


def success_first(rows):
    """Return SR, conditional successful time, and a combined score.

    Each row must contain ``success`` and ``deadline``. Failed rows must carry
    ``completion_ticks=None``; their duration is never imputed or averaged.
    """
    rows = list(rows)
    unassessed = [r for r in rows if r.get("evaluation_status", "evaluated") != "evaluated"]
    evaluated = [r for r in rows if r.get("evaluation_status", "evaluated") == "evaluated"]
    if any(not isinstance(r.get("success"), bool) for r in evaluated):
        raise ValueError("evaluated runs require boolean success")
    successes = [r for r in evaluated if bool(r["success"])]
    failures = [r for r in evaluated if not bool(r["success"])]
    if any(r.get("completion_ticks") is not None for r in failures):
        raise ValueError("failed runs must not have completion_ticks")
    times = [r["completion_ticks"] for r in successes]
    if any(t is None or t < 0 for t in times):
        raise ValueError("successful runs require completion_ticks")
    normalized = []
    for row in successes:
        deadline = row.get("deadline")
        if deadline is None or deadline <= 0:
            raise ValueError("successful runs require a positive deadline")
        normalized.append(min(1.0, max(0.0, row["completion_ticks"] / deadline)))
    sr = len(successes) / len(evaluated) if evaluated else None
    cond_time = mean(times) if times else None
    cond_norm = mean(normalized) if normalized else None
    time_score = 1.0 - cond_norm if cond_norm is not None else 0.0
    return {
        "sr": sr,
        "successes": len(successes),
        "runs": len(rows),
        "evaluated_runs": len(evaluated),
        "unassessed_runs": len(unassessed),
        "successful_mean_ticks": cond_time,
        "successful_median_ticks": median(times) if times else None,
        "successful_mean_normalized_time": cond_norm,
        "successful_time_score": time_score,
        "composite_score": sr * time_score if sr is not None else None,
        "failed_runs_have_no_time": all(r.get("completion_ticks") is None for r in failures),
    }
