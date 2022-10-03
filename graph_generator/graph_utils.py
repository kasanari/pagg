import yaml
from typing import TextIO

import networkx as nx

from .constants import *
from .graph import AttackGraph, InstanceModel

reward_mappings = {
    0: REWARD_DEFAULT,
    1: REWARD_EASY,
    2: REWARD_MEDIUM,
    3: REWARD_HARD,
}

def save_to_file(attack_graph: AttackGraph, instance_model: InstanceModel, file_pointer: TextIO):
    ag = [
        {
            "step_type": step.step_type,
            "ttc": step.ttc,
            "reward": reward_mappings[step.reward] if step.step_type != DEFENSE else step.reward,
            "children": [node_to_string(attack_graph, instance_model, s) for s in attack_graph.children(step.id)],
            "asset": instance_model.get_asset_type(attack_graph[step.id].asset),
            "id": node_to_string(attack_graph, instance_model, step.id),
            "name": step.id if step.step_name == "" else step.step_name,
        }
        for step in attack_graph
    ]

    im = [
        {
            "id": asset.id,
            "dependents": [a for a in nx.descendants(instance_model.graph, asset.id)]
        }
        for asset in instance_model
    ]

    flags = {
        node_to_string(attack_graph, instance_model, step.id) : reward_mappings[step.reward] for step in attack_graph if step.is_flag
    }

    to_save = {
        "attack_graph": ag,
        "instance_model": im,
        "flags": flags
    }

    yaml.dump(to_save, file_pointer)


def node_to_string(attack_graph: AttackGraph, instance_model: InstanceModel, node):
    asset_str = instance_model.asset_to_string(attack_graph[node].asset)
    
    node_name = attack_graph[node].step_name

    if attack_graph[node].step_type == DEFENSE:
        node_name = "defend"
    elif attack_graph[node].is_flag:
        node_name = "take"

    return f"{asset_str}:{node_name}:{node}"

def get_all_attack_steps_for_asset(attack_graph: AttackGraph, instance_model: InstanceModel, asset: int):
    return {node.id for node in attack_graph if node.asset == asset}.union(
        *[
            get_all_attack_steps_for_asset(attack_graph, instance_model, descendant)
            for descendant in nx.descendants(instance_model.graph, asset)
        ]
    )


def assign_host_assets(graph: nx.DiGraph, instance_model: InstanceModel, entrypoint):
    """Has to be done before lateral edges are added"""
    children = graph.successors(entrypoint)

    mappings = {}
    conditions = {}
    for c in children:
        new_asset = instance_model.add_asset(obj_class="server")
        mappings[c] = new_asset
        conditions[c] = {new_asset}
        for d in nx.descendants(graph, c):
            mappings[d] = new_asset
            conditions[d] = {new_asset}

    nx.set_node_attributes(graph, mappings, name=ASSET)
    nx.set_node_attributes(graph, conditions, name=CONDITIONS)
    return graph, instance_model


def attach_defense_step_to_graph(attack_graph: AttackGraph, defense_step, attack_steps, asset_id):

    old_parents = (list(attack_graph.graph.predecessors(step)) for step in attack_steps)

    for step, parents in zip(attack_steps, old_parents):
        # For OR steps we have to add to additional attack steps to preserve the logic
        if attack_graph[step].step_type == OR:
            # Create new attack steps
            new_AND_step = attack_graph.add_step(step_type=AND, parent=defense_step, ttc=0, asset=asset_id)
            new_OR_step = attack_graph.add_step(step_type=OR, ttc=0, asset=asset_id)
            # Connect OR-step and step to AND-step
            attack_graph.graph.add_edges_from([(new_AND_step, step), (new_OR_step, new_AND_step)])
            # Connect parents to OR-step
            for p in parents:
                attack_graph.graph.add_edge(p, new_OR_step)
                attack_graph.graph.remove_edge(p, step)
        # For AND steps we can simply add the defense step as a parent
        elif attack_graph[step].step_type == AND:
            attack_graph.graph.add_edge(defense_step, step)
        else:
            continue
