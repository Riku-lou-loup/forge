"""Point and event metrics with causal alerts, explicit gaps, and no point adjustment."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def causal_alerts(scores, timestamps, threshold, persistence=3, max_gap_seconds=2):
    if persistence < 1 or max_gap_seconds <= 0 or not np.isfinite(threshold):
        raise ValueError("Invalid alert policy.")
    scores = np.asarray(scores, dtype=float)
    times = pd.DatetimeIndex(timestamps)
    if (
        len(scores) != len(times)
        or times.hasnans
        or not times.is_monotonic_increasing
        or times.has_duplicates
    ):
        raise ValueError("Scores need unique, increasing timestamps of the same length.")
    alerts = np.zeros(len(scores), dtype=bool)
    run = 0
    for i, score in enumerate(scores):
        if i and (times[i] - times[i - 1]).total_seconds() > max_gap_seconds:
            run = 0
        run = run + 1 if np.isfinite(score) and score > threshold else 0
        alerts[i] = run >= persistence
    return alerts


def starts(mask, times, max_gap_seconds=2):
    mask = np.asarray(mask, dtype=bool)
    breaks = (
        pd.Series(pd.DatetimeIndex(times)).diff().dt.total_seconds().gt(max_gap_seconds).to_numpy()
    )
    previous = np.r_[False, mask[:-1]]
    return np.flatnonzero(mask & (~previous | breaks))


def evaluate_group(frame, scores, threshold, policy):
    labels = frame.anomaly.to_numpy(dtype=int)
    valid = labels >= 0
    # Unknown annotations break the evaluation timeline instead of being guessed.
    masked_scores = np.where(valid, scores, np.nan)
    alerts = causal_alerts(masked_scores, frame.datetime, threshold, **policy)
    y, predicted = labels[valid], alerts[valid]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="binary", zero_division=0
    )
    events = starts(labels == 1, frame.datetime, policy["max_gap_seconds"])
    onsets = starts(alerts, frame.datetime, policy["max_gap_seconds"])
    delays = []
    times = pd.DatetimeIndex(frame.datetime)
    for begin in events:
        end = begin + 1
        while (
            end < len(labels)
            and labels[end] == 1
            and (times[end] - times[end - 1]).total_seconds() <= policy["max_gap_seconds"]
        ):
            end += 1
        hits = onsets[(onsets >= begin) & (onsets < end)]
        if len(hits):
            delays.append((times[hits[0]] - times[begin]).total_seconds())
    intervals = np.r_[np.diff(times.asi8) / 1e9, 1.0]
    normal_hours = float(np.minimum(intervals, 1.0)[labels == 0].sum() / 3600)
    false_onsets = int((labels[onsets] == 0).sum())
    both_classes = len(np.unique(y)) == 2
    return {
        "rows": int(valid.sum()),
        "positive_rows": int((y == 1).sum()),
        "tp": int(((y == 1) & predicted).sum()),
        "fp": int(((y == 0) & predicted).sum()),
        "fn": int(((y == 1) & ~predicted).sum()),
        "tn": int(((y == 0) & ~predicted).sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": float(average_precision_score(y, np.asarray(scores)[valid]))
        if both_classes
        else None,
        "roc_auc": float(roc_auc_score(y, np.asarray(scores)[valid])) if both_classes else None,
        "events": len(events),
        "detected_events": len(delays),
        "event_recall": len(delays) / len(events) if len(events) else None,
        "detected_event_delays_seconds": delays,
        "false_alarm_onsets": false_onsets,
        "normal_hours": normal_hours,
        "false_alarms_per_normal_hour": false_onsets / normal_hours if normal_hours else None,
    }


def evaluate(groups, scores, threshold, policy):
    per_group = {
        name: evaluate_group(frame, scores[name], threshold, policy)
        for name, frame in groups.items()
    }
    summed = {
        key: sum(m[key] for m in per_group.values())
        for key in (
            "rows",
            "positive_rows",
            "tp",
            "fp",
            "fn",
            "tn",
            "events",
            "detected_events",
            "false_alarm_onsets",
            "normal_hours",
        )
    }
    tp, fp, fn = (summed[k] for k in ("tp", "fp", "fn"))
    summed.update(
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        event_recall=summed["detected_events"] / summed["events"] if summed["events"] else None,
        false_alarms_per_normal_hour=summed["false_alarm_onsets"] / summed["normal_hours"]
        if summed["normal_hours"]
        else None,
    )
    labels = np.concatenate([f.anomaly.to_numpy() for f in groups.values()])
    values = np.concatenate([scores[name] for name in groups])
    valid = labels >= 0
    summed["average_precision"] = (
        float(average_precision_score(labels[valid], values[valid]))
        if len(np.unique(labels[valid])) == 2
        else None
    )
    aps = [m["average_precision"] for m in per_group.values() if m["average_precision"] is not None]
    summed["macro_group_average_precision"] = float(np.mean(aps)) if aps else None
    delays = [d for m in per_group.values() for d in m["detected_event_delays_seconds"]]
    summed["median_detected_event_delay_seconds"] = float(np.median(delays)) if delays else None
    return {"pooled": summed, "per_group": per_group}


def select_threshold(groups, scores, policy, quantiles=101):
    values = np.concatenate(
        [scores[name][frame.anomaly.to_numpy() >= 0] for name, frame in groups.items()]
    )
    candidates = np.unique(
        np.r_[
            np.nextafter(values.min(), -np.inf), np.quantile(values, np.linspace(0, 1, quantiles))
        ]
    )
    best = None
    for threshold in candidates:
        result = evaluate(groups, scores, float(threshold), policy)
        pooled = result["pooled"]
        key = (pooled["f1"], -pooled["false_alarm_onsets"], float(threshold))
        if best is None or key > best[0]:
            best = key, float(threshold), result
    return best[1], best[2]
