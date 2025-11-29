#!/usr/bin/env python3
"""Plot concept confidence and belief divergence metrics from logs.

The script expects a metrics file (JSON or JSONL) produced during planner runs
containing entries like::

    {"step": 3, "avg_concept_confidence": 0.42,
     "divergence_metrics": {"concept_js_divergence": 0.31}}

Any dictionary entry with a ``divergence_metrics`` field will be parsed. If the
file is a directory, the script will search for the first ``*.jsonl`` or
``*.json`` file inside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def _discover_metrics_file(path: Path) -> Path:
    if path.is_file():
        return path
    for suffix in ("*.jsonl", "*.json", "*.yaml", "*.yml"):
        candidates = sorted(path.glob(suffix))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Could not find a metrics file under {path}")


def _load_records(path: Path) -> List[Dict]:
    if path.suffix == ".jsonl":
        records = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

    if path.suffix in {".yaml", ".yml"}:
        content = yaml.safe_load(path.read_text())
        return content if isinstance(content, list) else [content]

    content = json.loads(path.read_text())
    return content if isinstance(content, list) else [content]


def _mean_confidence(value: Union[Dict, Iterable, float, int, None]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict):
        values = list(value.values())
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        return float(value)

    if not values:
        return None
    return float(sum(values) / len(values))


def _extract_divergence(entry: Dict, divergence_key: str) -> Optional[float]:
    divergence_value = entry.get("belief_divergence") or entry.get("divergence")
    divergence_metrics = (
        entry.get("divergence_metrics")
        or entry.get("belief_divergence_metrics")
        or entry.get("divergence_dict")
    )
    if isinstance(divergence_metrics, dict):
        divergence_value = divergence_metrics.get(divergence_key, divergence_value)
        if divergence_value is None and divergence_metrics:
            divergence_value = next(iter(divergence_metrics.values()))
    elif isinstance(divergence_value, dict):
        divergence_value = divergence_value.get(divergence_key)
    return float(divergence_value) if divergence_value is not None else None


def _extract_series(records: List[Dict], divergence_key: str) -> Tuple[List[int], List[float], List[float]]:
    steps: List[int] = []
    confidences: List[float] = []
    divergences: List[float] = []
    for idx, entry in enumerate(records):
        if not isinstance(entry, dict):
            continue
        confidence_value = entry.get("avg_concept_confidence") or entry.get(
            "concept_confidence"
        )
        confidence = _mean_confidence(confidence_value)
        divergence = _extract_divergence(entry, divergence_key)
        if confidence is None and divergence is None:
            continue
        steps.append(int(entry.get("step", idx)))
        confidences.append(confidence if confidence is not None else float("nan"))
        divergences.append(divergence if divergence is not None else float("nan"))
    return steps, confidences, divergences


def plot_metrics(
    steps: List[int],
    confidences: List[float],
    divergences: List[float],
    divergence_key: str,
    concept_threshold: Optional[float],
    divergence_threshold: Optional[float],
    l2d_threshold: Optional[float],
    output_path: Path,
) -> None:
    fig, (conf_ax, div_ax) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    conf_ax.plot(steps, confidences, label="Avg concept confidence", color="C0")
    if concept_threshold is not None:
        conf_ax.axhline(concept_threshold, linestyle="--", color="C2", label="CBWM threshold")
    conf_ax.set_ylabel("Confidence")
    conf_ax.legend()
    conf_ax.grid(True, linestyle=":", alpha=0.5)

    div_ax.plot(steps, divergences, label=divergence_key, color="C1")
    if divergence_threshold is not None:
        div_ax.axhline(divergence_threshold, linestyle="--", color="C3", label="Divergence threshold")
    if l2d_threshold is not None:
        div_ax.axhline(l2d_threshold, linestyle=":", color="C4", label="L2D threshold")
    div_ax.set_xlabel("Step")
    div_ax.set_ylabel("Divergence")
    div_ax.legend()
    div_ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path, required=True, help="Metrics JSON/JSONL file or directory containing one.")
    parser.add_argument("--divergence-key", default="concept_js_divergence", help="Key to extract from divergence metrics.")
    parser.add_argument("--concept-threshold", type=float, default=None, help="Optional CBWM confidence threshold line.")
    parser.add_argument("--divergence-threshold", type=float, default=None, help="Optional divergence threshold line.")
    parser.add_argument("--l2d-threshold", type=float, default=None, help="Optional Listen-to-Disambiguate threshold line.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the figure. Defaults to <log-path>/belief_divergence.png",
    )
    args = parser.parse_args()

    metrics_path = _discover_metrics_file(args.log_path)
    records = _load_records(metrics_path)
    steps, confidences, divergences = _extract_series(records, args.divergence_key)

    if not steps:
        raise RuntimeError(f"No belief metrics found in {metrics_path}")

    output_path = args.output or metrics_path.with_name("belief_divergence.png")
    plot_metrics(
        steps,
        confidences,
        divergences,
        args.divergence_key,
        args.concept_threshold,
        args.divergence_threshold,
        args.l2d_threshold,
        output_path,
    )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
