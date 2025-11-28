#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""Utility functions to measure divergence between robot and human belief graphs."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, Tuple

from habitat_llm.world_model.world_graph import WorldGraph


def _distribution_from_counter(counter: Counter) -> Dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: val / total for key, val in counter.items()}


def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    eps = 1e-9
    divergence = 0.0
    for key, p_val in p.items():
        q_val = q.get(key, eps)
        divergence += p_val * math.log((p_val + eps) / (q_val + eps))
    return divergence


def _jensen_shannon(p: Dict[str, float], q: Dict[str, float]) -> float:
    if not p and not q:
        return 0.0
    all_keys = set(p.keys()) | set(q.keys())
    m = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in all_keys}
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _entity_confidence_gap(robot_graph: WorldGraph, human_graph: WorldGraph) -> float:
    """Compute average absolute difference in per-entity confidence."""

    robot_conf = {node.name: node.properties.get("confidence", 1.0) for node in robot_graph.graph}
    human_conf = {node.name: node.properties.get("confidence", 1.0) for node in human_graph.graph}
    all_entities = set(robot_conf) | set(human_conf)
    if not all_entities:
        return 0.0
    total_gap = 0.0
    for ent in all_entities:
        total_gap += abs(robot_conf.get(ent, 0.0) - human_conf.get(ent, 0.0))
    return total_gap / len(all_entities)


def _relation_consistency(robot_graph: WorldGraph, human_graph: WorldGraph) -> float:
    """Measure how many relations overlap across graphs."""

    def _edges(graph: WorldGraph) -> Iterable[Tuple[str, str, str]]:
        for node, neighbors in graph.graph.items():
            for neighbor, edge in neighbors.items():
                yield (node.name, neighbor.name, edge)

    robot_edges = set(_edges(robot_graph))
    human_edges = set(_edges(human_graph))
    if not robot_edges and not human_edges:
        return 1.0
    if not robot_edges or not human_edges:
        return 0.0
    intersection = robot_edges & human_edges
    union = robot_edges | human_edges
    return len(intersection) / len(union)


def compute_belief_divergence(
    robot_graph: WorldGraph, human_graph: WorldGraph
) -> Dict[str, float]:
    """Compute divergence metrics across robot and human belief graphs."""

    robot_type_counts = Counter([node.properties.get("type", "unknown") for node in robot_graph.graph])
    human_type_counts = Counter([node.properties.get("type", "unknown") for node in human_graph.graph])

    robot_distribution = _distribution_from_counter(robot_type_counts)
    human_distribution = _distribution_from_counter(human_type_counts)

    return {
        "concept_js_divergence": _jensen_shannon(robot_distribution, human_distribution),
        "entity_confidence_gap": _entity_confidence_gap(robot_graph, human_graph),
        "relation_consistency": _relation_consistency(robot_graph, human_graph),
    }

