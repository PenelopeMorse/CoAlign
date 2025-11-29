#!/usr/bin/env python3

"""
Lightweight helpers for routing planner decisions based on world model
confidence and belief divergence.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class BeliefMetrics:
    avg_concept_confidence: float = 1.0
    belief_divergence: float = 0.0
    divergence_metric: str = "belief_divergence"
    divergence_metrics: Optional[Dict[str, float]] = None
    note: str = ""


def choose_belief_action(decision_conf, metrics: BeliefMetrics) -> Tuple[Optional[str], str]:
    """Return the tool name and a short reason when a hook should run.

    The decision hierarchy is:
    1) If divergence is above the correction threshold, prefer correction.
    2) If average concept confidence is below the configured threshold, add
       more observations.
    3) If divergence is above the warning threshold, ask the human for help.
    """

    if decision_conf is None:
        return None, ""

    div_threshold = decision_conf.get("divergence_threshold", 0.3)
    correction_threshold = decision_conf.get("correction_divergence_threshold", div_threshold * 1.5)
    confidence_threshold = decision_conf.get("concept_confidence_threshold", 0.5)
    cbwm_enabled = decision_conf.get("enable_cbwm", decision_conf.get("cbwm_enabled", True))

    l2d_conf = decision_conf.get("l2d_action", {}) or {}
    l2d_enabled = l2d_conf.get("enable", False)
    l2d_threshold = l2d_conf.get("divergence_threshold", div_threshold)
    l2d_action = l2d_conf.get("action", "ListenToDisambiguate")

    divergence_value = float(metrics.belief_divergence or 0.0)
    divergence_metric = metrics.divergence_metric or "belief_divergence"

    if divergence_value >= correction_threshold:
        action = decision_conf.get("correction_action", "CorrectHuman")
        reason = (
            f"Belief divergence ({divergence_metric}) {divergence_value:.2f} exceeds correction "
            f"threshold {correction_threshold:.2f}."
        )
        return action, reason

    if cbwm_enabled and metrics.avg_concept_confidence < confidence_threshold:
        action = decision_conf.get("low_confidence_action", "AppendObservation")
        reason = (
            f"Average concept confidence {metrics.avg_concept_confidence:.2f} is below "
            f"threshold {confidence_threshold:.2f}."
        )
        return action, reason

    if l2d_enabled and divergence_value >= l2d_threshold:
        reason = (
            f"Belief divergence ({divergence_metric}) {divergence_value:.2f} exceeds "
            f"L2D threshold {l2d_threshold:.2f}."
        )
        return l2d_action, reason

    if divergence_value >= div_threshold:
        action = decision_conf.get("high_divergence_action", "AskHuman")
        reason = (
            f"Belief divergence ({divergence_metric}) {divergence_value:.2f} exceeds "
            f"threshold {div_threshold:.2f}."
        )
        return action, reason

    return None, metrics.note
