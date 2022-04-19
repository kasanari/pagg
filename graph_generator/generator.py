from itertools import permutations
import json
import logging
import random
from typing import Tuple

import networkx as nx
import numpy as np
from networkx.algorithms.shortest_paths.generic import shortest_path
from numpy.random import Generator, default_rng

from . import utils
from .constants import *
from .drawing import draw_attack_graph, draw_instance_model
from .utils import get_all_attack_steps_for_asset
from .graph import InstanceModel, AttackGraph


class GraphGenerator:
    def __init__(self, seed, max_nodes):
        self.seed = seed
        self.rng: Generator = default_rng(seed)
        self.max_nodes = max_nodes
        self.entrypoint = 0
        self.graph = AttackGraph()
        self.instance_model = InstanceModel()
        self.logger = logging.getLogger("generator")

    def select_next(self):
        nodes = self.graph.steps
        return int(self.rng.choice(nodes))

    def generate_initial_graph(self):
        # Create root step
        self.graph.add_step(
            step_type=self.rng.choice([AND, OR]),
            asset=self.instance_model.add_asset(obj_class="internet", unmalleable=True),
        )
        return [
            self.graph.add_step(step_type=self.rng.choice([AND, OR]), parent=self.select_next())
            for _ in range(1, self.max_nodes)
        ]

    def create_lateral_connections(self, num_connections):
        graph: AttackGraph = self.graph
        all_nodes = set(graph.steps)
        eligible_to_connect = list(
            filter(lambda node: not (node == self.entrypoint or graph.graph.out_degree(node) == 0), graph.steps)
        )

        if len(eligible_to_connect) < num_connections:
            self.logger.warning("Number of lateral connections requested is greater than the number of valid nodes.")
            num_connections = len(eligible_to_connect)

        nodes_to_add_connections_from = self.rng.choice(eligible_to_connect, size=num_connections, replace=False)

        def get_new_connection(node):
            descendants = nx.descendants(graph.graph, node)
            potential_connections = list(all_nodes - descendants - {node, self.entrypoint})
            if len(potential_connections) == 0:
                raise RuntimeError(
                    "Can not find any viable nodes to connect laterally!"
                    "Either add more nodes or decrease the number of lateral connections."
                )
            new_child = self.rng.choice(potential_connections)
            self.logger.info("Added edge %d -> %d", new_child, node)
            self.graph.graph.add_edge(int(new_child), int(node))
            return (int(new_child), int(node))

        return list(map(get_new_connection, nodes_to_add_connections_from))

    def assign_assets(self):
        """Has to be done before lateral edges are added"""

        # Give the children of the root node different assets
        assets = {c: self.instance_model.add_asset(obj_class="server") for c in self.graph.children(self.entrypoint)}

        # Update the descendants of the children with the new assets
        assets |= {d: assets[c] for c in assets for d in nx.descendants(self.graph.graph, c)}

        # Add the new assets as a condition
        conditions = {key: {value} for key, value in assets.items()}

        nx.set_node_attributes(self.graph.graph, assets, name=ASSET)
        nx.set_node_attributes(self.graph.graph, conditions, name=CONDITIONS)

    def set_flags(self, num_flags):

        if num_flags == 0:
            return 0

        nodes = self.graph.steps
        self.rng.shuffle(nodes)

        eligible_nodes = list(filter(lambda node: self.graph.graph.out_degree(node) == 0, nodes))

        if len(eligible_nodes) < num_flags:
            self.logger.warning("Number of flags requested is higher than the number of eligible nodes.")
            num_flags = len(eligible_nodes)
        elif len(eligible_nodes) == 0:
            self.logger.warning("No nodes in graph are eligible to become flags.")
            return 0

        flags = self.rng.choice(eligible_nodes, size=num_flags, replace=False)

        distances = {f: len(shortest_path(self.graph.graph, self.entrypoint, f)) for f in flags}

        d = np.array(list(distances.values()))
        medium_bar = np.quantile(d, 1 / 3)
        hard_bar = np.quantile(d, 2 / 3)

        def get_reward(distance):
            if distance <= medium_bar:
                return REWARD_EASY
            if distance <= hard_bar:
                return REWARD_MEDIUM
            return REWARD_HARD

        flags_with_rewards = {f: get_reward(distance) for f, distance in distances.items()}

        new_assets = {f: self.instance_model.add_asset(self.graph[f].asset, flag=True, unmalleable=True) for f in flags}

        nx.set_node_attributes(self.graph.graph, new_assets, ASSET)
        nx.set_node_attributes(self.graph.graph, flags_with_rewards, "reward")
        return len(flags)

    def node_to_string(self, node):
        utils.node_to_string(self.graph, self.instance_model, node)

    def add_defense_steps(self, attack_graph: AttackGraph, instance_model: InstanceModel):
        assets = list(instance_model)
        defense_asset = instance_model.add_asset(unmalleable=True)
        defense_steps = [
            (
                attack_graph.add_step(
                    step_type=DEFENSE,
                    asset=defense_asset,
                    reward=self.instance_model.calculate_defense_cost(asset.id),
                    assets_disabled=[descendant for descendant in nx.descendants(self.instance_model.graph, asset.id)],
                ),
                get_all_attack_steps_for_asset(attack_graph, instance_model, asset.id),
                asset.id,
            )
            for asset in assets
            if not asset.unmalleable
        ]

        for step, attack_steps, asset_id in defense_steps:
            utils.attach_defense_step_to_graph(attack_graph, step, attack_steps, asset_id)

        return defense_steps

    def set_step_as_new_asset(self, node):
        new_asset = self.instance_model.add_asset(parent=self.graph[node].asset, obj_class="host")
        self.graph.update_step(node, new_asset)
        descendants = list(nx.descendants(self.graph.graph, node))
        return [self.graph.update_step(n, new_asset) for n in descendants]

    def add_more_assets(self):
        # high degree
        # parent is from same asset
        # many descendants with same asset as parent

        def few_in_many_out(node, _):
            return self.graph.graph.out_degree(node) > 5 and self.graph.graph.in_degree(node) == 1

        def many_descendants(_, descendants):
            return len(descendants) > 5

        conditions = [few_in_many_out, many_descendants]
        selected_nodes = set()

        for step in self.graph:
            try:
                parent = self.graph.children(step.id).__next__()
            except StopIteration:
                continue
            parent_asset = self.graph[parent].asset
            if step.asset == parent_asset:
                descendants = nx.descendants(self.graph.graph, step.id)
                for func in conditions:
                    if func(step.id, descendants):
                        selected_nodes.add(step.id)

        for node in selected_nodes:
            self.set_step_as_new_asset(node)

    @classmethod
    def generate_attack_graph(
        cls, seed, size, lateral_connections, num_flags, add_defense_steps=True
    ) -> Tuple[AttackGraph, InstanceModel]:

        generator = cls(seed, size)
        generator.generate_initial_graph()
        generator.assign_assets()
        generator.add_more_assets()

        if lateral_connections > 0:
            new_connections = generator.create_lateral_connections(lateral_connections)
            generator.logger.info(
                "Added %d out of %d requested lateral connections.", len(new_connections), lateral_connections
            )

        if num_flags > 0:
            generator.set_flags(num_flags=num_flags)

        if add_defense_steps:
            generator.add_defense_steps(generator.graph, generator.instance_model)

        return (generator.graph, generator.instance_model)


def generate_graph(name, size, lateral_connections, num_flags, seed):

    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger("main")

    random.seed(seed)

    attack_graph, instance_model = GraphGenerator.generate_attack_graph(
        seed, size, lateral_connections, num_flags, add_defense_steps=True
    )

    reachable, acyclic = utils.validate_graph(attack_graph.graph, attack_graph.entrypoint)

    if not reachable:
        logger.warning("Not all AND steps are reachable from entrypoint.")

    if not acyclic:
        logger.warning("Graph is not acyclic.")

    avg_cc = np.mean(list(nx.clustering(attack_graph.graph).values()))
    logger.info("Average CC: %d", avg_cc)

    mean_in_degress = np.mean(attack_graph.graph.in_degree())
    mean_out_degrees = np.mean(attack_graph.graph.out_degree())

    with open(f"model_{name}.json", "w", encoding="utf-8") as f:
        utils.save_to_file(attack_graph, instance_model, f)

    logger.info("Nodes have %f incoming edges, and %f outgoing edges on average.", mean_in_degress, mean_out_degrees)

    lengths = []

    for i, j in permutations(attack_graph.steps, 2):
        try:
            paths = nx.all_shortest_paths(attack_graph.graph, i, j)
            lengths.extend([len(p) for p in paths])
        except nx.NetworkXNoPath:
            continue

    logger.info("Graph has %d shortest paths with an average length of %f nodes. ", len(lengths), np.mean(lengths))

    # Count assets
    assets = set()
    for asset in nx.get_node_attributes(attack_graph.graph, ASSET).values():
        assets.add(asset)

    logger.info("Graph contains %d steps with %d assets.", attack_graph.step_count, len(assets))

    logger.info("Graph has %d flags.", num_flags)

    draw_attack_graph(attack_graph, seed, f"ag_{name}.pdf")

    draw_instance_model(instance_model.graph, f"instance_{name}.pdf")

    conditions = nx.get_node_attributes(attack_graph.graph, CONDITIONS)
    conditions = {k: list(v) for k, v in conditions.items()}
    nx.set_node_attributes(attack_graph.graph, conditions, CONDITIONS)

    with open(f"nx_{name}.json", "w", encoding="utf8") as f:
        json.dump(nx.node_link_data(attack_graph.graph), f)
