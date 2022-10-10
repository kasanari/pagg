from itertools import permutations
import logging
from pathlib import Path
import random
from typing import Dict, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms.shortest_paths.generic import shortest_path
from numpy.random import Generator, default_rng
import scipy.stats as stats

from .constants import ASSET, CONDITIONS, REWARD, TTC, STEP

from .drawing import draw_attack_graph
from .graph_utils import get_all_attack_steps_for_asset
from .graph import  InstanceModel, AttackGraph
from graph_generator import graph_utils, nx_utils
import enum

class Mode(enum.Enum):
    RANDOM = "random"
    FITNESS = "fitness"
    BIANCONI = "bianconi"
    DEGREE = "degree"

class GraphGenerator:
    def __init__(self, seed, max_nodes, mode=Mode.RANDOM, mu=0):
        self.seed = seed
        self.rng: Generator = default_rng(seed)
        self.max_nodes = max_nodes
        self.entrypoint = 0
        self.graph = AttackGraph()
        self.instance_model = InstanceModel()
        self.logger = logging.getLogger("generator")
        #self.fitness = stats.binom.pmf(np.arange(max_nodes), n=max_nodes, p=mu / max_nodes)
        self.fitness = stats.truncnorm.pdf(np.arange(max_nodes), a=0, b=max_nodes, loc=0, scale=mu)

        self.mu = mu
        self.mode = mode

    def plot_fitness_distribution(self):
        import matplotlib.pyplot as plt

        plt.plot(self.fitness)
        plt.savefig(f"fitness_{self.mu/self.max_nodes}.pdf")
        plt.close()

    def select_next(self):
        nodes = self.graph.steps

        degrees = np.array([degree for _, degree in self.graph.graph.degree()])
        fitness = self.fitness[: len(nodes)]
        if len(degrees) == 1 or self.mode == Mode.RANDOM:
            weights = None
        elif self.mode == Mode.BIANCONI:
            weights = (degrees * fitness) / np.sum(degrees * fitness)
        elif self.mode == Mode.DEGREE:
            weights = (degrees) / np.sum(degrees)
        elif self.mode == Mode.FITNESS:
            weights = (fitness) / np.sum(fitness)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return int(self.rng.choice(nodes, p=weights))

    def generate_initial_graph(self):
        # Create root step
        root_asset = self.instance_model.add_asset(obj_class="internet", unmalleable=True)
        self.graph.add_step(
            step_type=STEP.OR,
            asset=root_asset,
        )
        return [
            self.graph.add_step(step_type=[STEP.AND, STEP.OR][self.rng.integers(0, 2)], parent=self.select_next(), asset=root_asset)
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
                return
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

    def set_flags(self, num_flags) -> Dict[int, REWARD]:
        """Set the flags for the graph. This is done after the graph is generated"""
        # These are just template indexes, not the actual reward values

        if num_flags == 0:
            return {}

        nodes = self.graph.steps
        self.rng.shuffle(nodes)

        def filter_func(node):
            return (
                self.graph.graph.out_degree(node) == 0
                and self.graph.graph.in_degree(node) > 0
                and self.entrypoint not in self.graph.graph.predecessors(node)
            )

        eligible_nodes = list(filter(filter_func, nodes))

        if len(eligible_nodes) < num_flags:
            self.logger.warning("Number of flags requested is higher than the number of eligible nodes.")
            num_flags = len(eligible_nodes)
        elif len(eligible_nodes) == 0:
            self.logger.warning("No nodes in graph are eligible to become flags.")
            return {}

        flags = self.rng.choice(eligible_nodes, size=num_flags, replace=False)

        distances = {f: len(shortest_path(self.graph.graph, self.entrypoint, f)) for f in flags}

        d = np.unique(np.array(list(distances.values())))
        medium_bar = np.quantile(d, 1 / 3)
        hard_bar = np.quantile(d, 2 / 3)

        def get_reward(distance):
            if distance <= medium_bar:
                return REWARD.EASY
            if distance <= hard_bar:
                return REWARD.MEDIUM
            return REWARD.HARD

        flags_with_rewards = {f: get_reward(distance) for f, distance in distances.items()}

        #new_assets = {f: self.instance_model.add_asset(self.graph[f].asset, flag=True, unmalleable=True) for f in flags}

        #nx.set_node_attributes(self.graph.graph, new_assets, ASSET)
        #nx.set_node_attributes(self.graph.graph, flags_with_rewards, "reward")
        return flags_with_rewards

    def add_defense_steps(self, attack_graph: AttackGraph, instance_model: InstanceModel, add_attack_steps=False, include_descendants=False):
        """
        Adds defense steps to the graph.
        :param attack_graph: The graph to add defense steps to.
        :param instance_model: The instance model to use for the defense steps.
        :param add_attack_steps: Whether to add additional attack steps to the graph.
        """
        all_assets = list(instance_model)
        assets = set(nx.get_node_attributes(attack_graph.graph, "asset").values())

        # defense_asset = instance_model.add_asset(unmalleable=True)
        defense_steps = [
            (
                attack_graph.add_step(
                    step_type=STEP.DEFENSE,
                    asset=None,
                    reward=self.instance_model.calculate_defense_cost(asset),
                    assets_disabled=list(nx.descendants(self.instance_model.graph, asset)) if include_descendants else [asset],
                ),
                get_all_attack_steps_for_asset(attack_graph, instance_model, asset, include_descendants),
                asset,
            )
            for asset in assets
            if not all_assets[asset].unmalleable
        ]

        if add_attack_steps:
            for step, attack_steps, asset_id in defense_steps:
                graph_utils.attach_defense_step_to_graph(attack_graph, step, attack_steps, asset_id)
        else:
            for step, attack_steps, asset_id in defense_steps:
                for attack_step in attack_steps:
                    attack_graph.graph.add_edge(step, attack_step)

        defense_assets = {node: asset for node, _, asset in defense_steps}
        nx.set_node_attributes(attack_graph.graph, defense_assets, name=ASSET)

        return defense_steps

    def set_step_as_new_asset(self, node):
        new_asset = self.instance_model.add_asset(parent=self.graph[node].asset, obj_class="host")
        self.graph.update_step(node, new_asset)
        descendants = list(nx.descendants(self.graph.graph, node))
        return [self.graph.update_step(n, new_asset) for n in descendants]

    def add_flag_defenses(self, flags):
        for flag in flags:
            flag_node = self.graph[flag]
            asset = flag_node.asset
            defense_step = self.graph.add_step(step_type=STEP.DEFENSE, ttc=TTC.DEFAULT, asset=asset)
            firewall = self.graph.add_step(step_type=STEP.AND, parent=defense_step, ttc=TTC.NONE, asset=asset)
            
            flag_parents = list(self.graph.graph.predecessors(flag))
            
            for parent in flag_parents:
                self.graph.graph.add_edge(parent, firewall)
            
            for parent in flag_parents:
                self.graph.graph.remove_edge(parent, flag)
            
            self.graph.graph.add_edge(firewall, flag)

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
        cls, seed, size, lateral_connections, num_flags, mode, mu, add_defense_steps=True
    ) -> Tuple[AttackGraph, InstanceModel]:

        generator = cls(seed, size, mode, mu)
        generator.generate_initial_graph()
        generator.assign_assets()
        generator.add_more_assets()

        if lateral_connections > 0:
            new_connections = generator.create_lateral_connections(lateral_connections)
            generator.logger.info(
                "Added %d out of %d requested lateral connections.", len(new_connections), lateral_connections
            )

        if num_flags > 0:
            flags = generator.set_flags(num_flags=num_flags)

        if add_defense_steps:
            #generator.add_defense_steps(generator.graph, generator.instance_model)
            generator.add_flag_defenses(flags)

        generator.plot_fitness_distribution()

        return (generator.graph, generator.instance_model, flags)


def generate_graph(name, size, lateral_connections, num_flags, add_defense, seed, mode, mu):

    log_level = logging.INFO

    graph_folder = Path("graphs")

    if not graph_folder.exists():
        graph_folder.mkdir()

    yaml_path = graph_folder / f"model_{name}.yaml"
    img_path = graph_folder / f"ag_{name}.pdf"

    logging.basicConfig(level=log_level)
    logger = logging.getLogger("main")

    random.seed(seed)

    attack_graph, instance_model, flags = GraphGenerator.generate_attack_graph(
        seed, size, lateral_connections, num_flags, mode, mu, add_defense_steps=add_defense
    )

    reachable, acyclic = nx_utils.validate_graph(attack_graph.graph, attack_graph.entrypoint)

    if not reachable:
        logger.warning("Not all AND steps are reachable from entrypoint.")

    if not acyclic:
        logger.warning("Graph is not acyclic.")

    avg_cc = np.mean(list(nx.clustering(attack_graph.graph).values()))
    logger.info("Average CC: %d", avg_cc)

    mean_in_degress = np.mean([v for _, v in attack_graph.graph.in_degree()])
    mean_out_degrees = np.mean([v for _, v in attack_graph.graph.out_degree()])

    with open(yaml_path, "w", encoding="utf-8") as f:
        graph_utils.save_to_file(attack_graph, instance_model, flags, f)

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

    draw_attack_graph(attack_graph, flags, seed, img_path)

    # draw_instance_model(instance_model.graph, f"instance_{name}.pdf")

    # conditions = nx.get_node_attributes(attack_graph.graph, CONDITIONS)
    # conditions = {k: list(v) for k, v in conditions.items()}
    # nx.set_node_attributes(attack_graph.graph, conditions, CONDITIONS)

    # with open(f"nx_{name}.json", "w", encoding="utf8") as f:
    #    json.dump(nx.node_link_data(attack_graph.graph), f)
