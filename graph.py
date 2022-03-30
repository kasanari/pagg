from audioop import add
from operator import itemgetter
from select import select
from matplotlib import scale
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.distance_measures import center
from networkx.algorithms.shortest_paths.unweighted import predecessor

from name_generator import NameGenerator
import random

from numpy.random import default_rng
import numpy as np

import logging

import yaml

import json

AND = "AND"
OR = "OR"

ASSET = "asset"
STEP_TYPE = "step_type"
CONDITIONS = "conditions"

TTC_EASY = "EASY_TTC"
TTC_HARD = "HARD_TTC"

REWARD_EASY = "LOW_FLAG_REWARD"
REWARD_MEDIUM = "MEDIUM_FLAG_REWARD"
REWARD_HARD = "HIGH_FLAG_REWARD"


def draw_graph(graph, seed, outfile=None, num_assets=None):

    lengths = list(nx.shortest_path_length(graph, source=0).values())
    node_colors = np.array(list(nx.get_node_attributes(graph, "asset").values()))

    #if num_assets is not None:
    #    node_colors = node_colors / num_assets

    node_size = 300
    node_options = {
        "node_size": node_size,
        "node_color": node_colors,
        "edgecolors": "black",
        "linewidths": 2,
    }

    edge_options = {"width": 2, "node_size": node_size}

    label_options = {
        "font_size": 9,
    }

    # pos = nx.kamada_kawai_layout(graph)
    # pos = nx.spring_layout(graph, k=10, iterations=1000, fixed=[0], pos={0:(0,0)}, center=(0,0), seed=seed)
    # pos = nx.planar_layout(graph, scale=10)
    # pos = nx.spectral_layout(graph)
    pos = nx.nx_pydot.graphviz_layout(graph, root=0, prog="sfdp")

    OR_edges = []
    AND_edges = []
    for p, c in graph.edges:
        child_type = graph.nodes[c][STEP_TYPE]

        if child_type == OR:
            OR_edges.append((p, c))
        elif child_type == AND:
            AND_edges.append((p, c))
        else:
            raise Exception(f"Invalid child type")

    plt.figure(figsize=(20, 20))
    nx.draw_networkx_edges(graph, pos, OR_edges, **edge_options)
    nx.draw_networkx_edges(graph, pos, AND_edges, style="dashed", **edge_options)
    nx.draw_networkx_nodes(
        graph, pos, **node_options, cmap=plt.cm.get_cmap('gist_ncar')
    )  # default spring_layout
    nx.draw_networkx_labels(graph, pos, **label_options)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)

    plt.close()


def draw_instance_model(graph, filename):
    node_size = 300

    node_colors = np.array(graph.nodes)

    #node_colors = node_colors / len(node_colors)

    options = {
        "node_size": node_size,
        "edgecolors": "black",
        "node_color": node_colors,
        "linewidths": 2,
        "width": 2, 
        "node_size": node_size,
        "font_size": 9,
    }

    pos = nx.nx_pydot.graphviz_layout(graph, prog="dot")

    nx.draw_networkx(graph, pos, **options, cmap=plt.cm.get_cmap('gist_ncar'))
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    plt.close()
class BaseGenerator:
    def __init__(self, seed, max_nodes) -> None:
        self.seed = seed
        self.rng = default_rng(seed)
        self.max_nodes = max_nodes
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger("generator")
class DirectedGraphGenerator(BaseGenerator):
    def __init__(self, seed, max_nodes):
        super().__init__(seed, max_nodes)
        self.instance_model = nx.DiGraph()
        self.asset_count = 0

    def select_next(self):
        nodes = list(self.graph.nodes)
        return self.rng.choice(nodes)

    def generate_graph(self):

        current_node = 0
        step_type = self.rng.choice([AND, OR])
        self.graph.add_node(current_node, step_type=step_type, asset=self.create_new_asset(), conditions={}, ttc=10)
        self.instance_model.add_node(0)
        for n in range(1, self.max_nodes):
            step_type = self.rng.choice([AND, OR])
            child = self.select_next()

            self.graph.add_node(n, step_type=step_type, ttc=10)
            self.graph.add_edge(child, n, ttc=10)

        return self.graph

    def create_lateral_connections(self, num_connections):

        all_nodes = set(list(self.graph.nodes()))
        to_connect = list(self.graph.nodes())
        to_connect.remove(0)

        for _ in range(num_connections):

            node = int(self.rng.choice(to_connect))
            to_connect.remove(node)

            descendants = nx.descendants(self.graph, node)

            if descendants != None:
                potential_connections = all_nodes.difference({node})
                potential_connections = list(
                    potential_connections.difference(descendants)
                )

                new_child = self.rng.choice(potential_connections)
                self.logger.info(f"Added edge {new_child} -> {node}")
                # self.graph.add_edge(node, new_child)
                self.graph.add_edge(new_child, node)
                # edges.append((node, self.rng.choice(potential_connections)))

            pass

        # self.graph.add_edges_from(edges)
        return self.graph

    def check_AND_reachability(self, entry_node):
        unreachable = []
        valid = True
        for node, attributes in self.graph.nodes().items():
            if attributes[STEP_TYPE] == AND:
                required_steps = self.graph.predecessors(node)
                for step in required_steps:
                    paths = nx.shortest_path(self.graph, source=entry_node, target=step)
                    if len(paths) == 0:
                        unreachable.append(node)
                        valid = False

        return valid, unreachable

    def set_edge_attributes(self):

        for node, attributes in self.graph.nodes().items():
            step_type = attributes[STEP_TYPE]
            for predecessor in self.graph.predecessors(node):
                self.graph.add_edge(predecessor, node, step_type=step_type)

    def assign_assets(self, entrypoint):
        """Has to be done before lateral edges are added"""
        children = self.graph.successors(entrypoint)

        mappings = {}
        conditions = {}
        for c in children:
            new_asset = self.create_new_asset()
            mappings[c] = new_asset
            conditions[c] = {new_asset}
            for d in nx.descendants(self.graph, c):
                mappings[d] = new_asset
                conditions[d] = {new_asset}

        nx.set_node_attributes(self.graph, mappings, name=ASSET)
        nx.set_node_attributes(self.graph, conditions, name=CONDITIONS)

    def set_flags(self, entrypoint, num_flags=None):
        flags = []
        flags_with_rewards = {}
        flags_added = 0
        nodes = list(self.graph.nodes)
        self.rng.shuffle(nodes)
        for n in self.graph.nodes:
            if self.graph.out_degree(n) == 0:
                flags.append(n)
                flags_added += 1
                if num_flags is not None and flags_added == num_flags:
                    break

        distances = {}
        for f in flags:
            distance_to_start = len(nx.shortest_path(self.graph, entrypoint, f))
            distances[f] = distance_to_start

        d = np.array(list(distances.values()))
        medium_bar = np.quantile(d, 1 / 3)
        hard_bar = np.quantile(d, 2 / 3)

        for f, d in distances.items():
            if d <= medium_bar:
                flags_with_rewards[f] = REWARD_EASY
            elif d <= hard_bar:
                flags_with_rewards[f] = REWARD_MEDIUM
            else:
                flags_with_rewards[f] = REWARD_HARD

        nx.set_node_attributes(self.graph, flags_with_rewards, "reward")
        return len(flags)

    def node_to_string(self, node):
        name = f"{self.graph.nodes[node]['asset']}.{node}"

        if "reward" in self.graph.nodes[node]:
            name += "_flag"

        return name

    def to_yaml(self, filename):
        to_write = {}
        for n, attributes in self.graph.nodes.items():
            name = self.node_to_string(n)
            data = {}
            if attributes[STEP_TYPE] == AND:
                data[STEP_TYPE] = "and"
            elif attributes[STEP_TYPE] == OR:
                data[STEP_TYPE] = "or"

            data["ttc"] = TTC_EASY

            if "reward" in attributes:
                data["reward"] = attributes["reward"]

            data["children"] = []
            for s in self.graph.successors(n):
                data["children"].append(self.node_to_string(s))

            data[CONDITIONS] = [a for a in attributes[CONDITIONS]]

            to_write[name] = data

        with open(f"ag_{filename}", "w") as f:
            yaml.dump(to_write, f)


        with open(f"instance_{filename}", "w") as f:
            dependents= {}
            for n in self.instance_model.nodes:
                dependents[str(n)] = []
                for d in nx.descendants(self.instance_model, n):
                    dependents[str(n)].append(str(d)) 

            yaml.dump(dependents, f)
                
    def create_new_asset(self, parent=None):
        new_asset = self.asset_count
        if parent is not None:
                self.instance_model.add_edge(parent, new_asset)
        else:
            self.instance_model.add_node(new_asset)
        self.asset_count += 1
        return new_asset


    def add_more_assets(self):
        # high degree
        # parent is from same asset
        # many descendants with same asset as parent

        def update_node(node, asset):
            # Set the asset of the node
            self.graph.nodes[node][ASSET] = asset
            # Add the old asset as a condition
            self.graph.nodes[node][CONDITIONS].add(old_asset)
            self.graph.nodes[node][CONDITIONS].add(asset)

        def node_has_high_out(node, descendants):
            if (
                self.graph.out_degree(node) > 5
                and self.graph.in_degree(node) == 1
            ):
                old_asset = self.graph.nodes[node]["asset"]
                new_asset = self.create_new_asset(old_asset)
                update_node(node, new_asset)
                for n in descendants:
                    update_node(n, new_asset)


        def node_has_many_children(node, descendants):
            if len(descendants) > 5:
                old_asset = self.graph.nodes[node]["asset"]
                new_asset = self.create_new_asset(old_asset)
                update_node(node, new_asset)
                for n in descendants:
                    update_node(n, new_asset)

        conditions = [node_has_high_out, node_has_many_children]
        for func in conditions:
            for node, attributes in self.graph.nodes.items():

                try:
                    parent = self.graph.successors(node).__next__()
                except StopIteration:
                    continue
                parent_asset = self.graph.nodes[parent]["asset"]
                descendants = nx.descendants(self.graph, node)
                asset = attributes[ASSET]
                if asset == parent_asset:
                    func(node, descendants)
                    pass


def add_unique(path: list, item):
    if item not in path:
        path.append(item)


class PathFinderAttacker:
    def __init__(self, attack_graph: nx.DiGraph, start_node) -> None:
        self.attack_graph: nx.DiGraph = attack_graph
        self.total_path = []
        self.start_node = start_node

    def find_path_to(self, target):
        path, _ = self._find_path_to(target)
        for p in path:
            add_unique(self.total_path, p)
        return self.total_path

    def _find_path_to(self, target):
        ttc_cost = 0

        path: list = nx.shortest_path(
            self.attack_graph, source=self.start_node, target=target, weight="ttc"
        )

        # Check each step in the path.
        for node_id in path:
            step = self.attack_graph.nodes[node_id]
            # If node is AND step, go to parents first.
            if step[STEP_TYPE] == AND and node_id not in self.total_path:
                parents = self.attack_graph.predecessors(node_id)
                paths_to_parents = []
                for p in parents:
                    # If the parent is already in the path, there is no need to find a path to it
                    if p not in self.total_path:
                        path_to_parent, cost = self._find_path_to(p)
                        paths_to_parents.append((path_to_parent, cost))

                for p, _ in sorted(paths_to_parents, key=itemgetter(1)):
                    [add_unique(self.total_path, n) for n in p]

            ttc_cost += step["ttc"]
            add_unique(self.total_path, node_id)

        return path, ttc_cost


def generate_graph(name, size, lateral_connections, num_flags, seed):

    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger("main")

    random.seed(seed)

    initial_step = 0

    graph_generator = DirectedGraphGenerator(seed, size)

    attack_graph = graph_generator.generate_graph()

    graph_generator.assign_assets(initial_step)

    # If an AND step only has one parent, consider it as an OR step
    # for node_key in attack_graph.nodes():
    #     node = attack_graph.nodes()[node_key]
    #     if node[STEP_TYPE] == AND and len(list(attack_graph.predecessors(node_key))) == 1:
    #         attack_graph.add_node(node_key, step_type=OR)

    valid, _ = graph_generator.check_AND_reachability(initial_step)

    if not valid:
        logger.warning(
            f"Not all AND steps are reachable from intitial step {initial_step}"
        )

    if not nx.is_directed_acyclic_graph(attack_graph):
        logger.warning(f"Graph is not acyclic.")

    graph_generator.set_edge_attributes()
    num_flags = graph_generator.set_flags(initial_step, num_flags=num_flags)

    graph_generator.add_more_assets()

    if lateral_connections > 0:
        graph_generator.create_lateral_connections(lateral_connections)

    cc = nx.clustering(attack_graph)
    avg_cc = np.mean(list(cc.values()))
    logger.info(f"Average CC: {avg_cc}")

    in_degress = attack_graph.in_degree()
    out_degrees = attack_graph.out_degree()

    graph_generator.to_yaml(f"{name}.yaml")

    avg_in = 0
    avg_out = 0

    for (_, in_d), (_, out_d) in zip(in_degress, out_degrees):
        avg_in += in_d
        avg_out += out_d

    avg_in /= len(in_degress)
    avg_out /= len(out_degrees)

    logger.info(
        f"Nodes have {avg_in} incoming edges, and {avg_out} outgoing edges on average."
    )

    num_paths = 0
    lengths = []

    for i in range(len(attack_graph.nodes())):
        for j in range(len(attack_graph.nodes())):
            paths = nx.all_shortest_paths(attack_graph, i, j)
            try:
                lengths.extend([len(p) for p in paths])
            except Exception:
                continue
            num_paths += len(list(paths))

    logger.info(
        f"The graph has {len(lengths)} shortest paths with an average length of {np.mean(lengths)} nodes. "
    )

    # count assets

    assets = set()
    for asset in nx.get_node_attributes(attack_graph, ASSET).values():
        assets.add(asset)

    logger.info(
        f"The graph contains {attack_graph.number_of_nodes()} steps with {len(assets)} assets."
    )

    logger.info("The graph has %d flags.", num_flags)

    # attacker = PathFinderAttacker(attack_graph, start_node=0)

    # path = attacker.find_path_to(72)

    # print(path)

    draw_graph(attack_graph, seed, f"ag_{name}.pdf", num_assets=graph_generator.instance_model.number_of_nodes())
    draw_instance_model(graph_generator.instance_model, f"instance_{name}.pdf")

    conditions = nx.get_node_attributes(attack_graph, CONDITIONS)
    conditions = {k : list(v) for k, v in conditions.items()}
    nx.set_node_attributes(attack_graph, conditions, CONDITIONS)

    with open(f"{name}.json", "w") as f:
        json.dump(nx.node_link_data(attack_graph), f)

    pass


def main(seed=888):

    configs = [
        dict(name="small", size=50, lateral_connections=5, num_flags=5,),
        dict(name="medium", size=100, lateral_connections=25, num_flags=10,),
        dict(name="big", size=200, lateral_connections=50, num_flags=20,),
    ]

    for config in configs:
        generate_graph(**config, seed=seed)


if __name__ == "__main__":
    main()

