#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import pytest

pytest.importorskip("habitat_sim")

from habitat_llm.world_model import BeliefGraphContainer, WorldGraph
from habitat_llm.world_model.belief_divergence import compute_belief_divergence
from habitat_llm.world_model import House, Object


def _make_graph(prefix: str) -> WorldGraph:
    graph = WorldGraph(graph_type=prefix)
    house = House("house", {"type": "root"}, "house")
    obj = Object(f"{prefix}_obj", {"type": "object", "translation": [0, 0, 0]}, prefix)
    graph.add_node(house)
    graph.add_node(obj)
    graph.add_edge(house, obj, "contains", "in")
    obj.properties["confidence"] = 0.9 if prefix == "robot" else 0.4
    return graph


def test_robot_update_does_not_touch_human_graph():
    robot_graph = WorldGraph(graph_type="robot")
    human_graph = WorldGraph(graph_type="human")
    container = BeliefGraphContainer(robot_graph=robot_graph, human_graph=human_graph)

    robot_view = _make_graph("robot")
    container.update_robot_belief(robot_view, partial_obs=False, update_mode="gt")

    assert container.get_graph("robot").has_node("robot_obj")
    assert container.get_graph("human").is_empty()


def test_compute_belief_divergence_metrics_present():
    robot_graph = _make_graph("robot")
    human_graph = _make_graph("human")
    # remove an edge from the human graph to reduce relation consistency
    human_graph.remove_edge("house", "human_obj")

    divergence = compute_belief_divergence(robot_graph, human_graph)

    assert set(divergence.keys()) == {
        "concept_js_divergence",
        "entity_confidence_gap",
        "relation_consistency",
    }
    assert divergence["entity_confidence_gap"] > 0
    assert 0 <= divergence["relation_consistency"] <= 1
