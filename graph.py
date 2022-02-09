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

TTC_EASY = "EASY_TTC"
TTC_HARD = "HARD_TTC"

REWARD_EASY = "LOW_FLAG_REWARD"
REWARD_MEDIUM = "MEDIUM_FLAG_REWARD"
REWARD_HARD = "HIGH_FLAG_REWARD"

def draw_graph(graph, seed, outfile=None):
    fig = plt.figure(figsize=(10, 10))

    lengths = list(nx.shortest_path_length(graph, source=0).values())
    node_colors = list(nx.get_node_attributes(graph, "asset").values())

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
    #pos = nx.spring_layout(graph, k=10, iterations=1000, fixed=[0], pos={0:(0,0)}, center=(0,0), seed=seed)
    # pos = nx.planar_layout(graph, scale=10)
    # pos = nx.spectral_layout(graph)
    pos = nx.nx_pydot.graphviz_layout(graph, root=0, prog="sfdp")

    
    OR_edges = []
    AND_edges = []
    for p, c in graph.edges:
        child_type = graph.nodes[c]["step_type"]

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
        graph, pos, **node_options, cmap=plt.cm.get_cmap("tab20")
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


class BaseGenerator:
    def __init__(self, seed, max_nodes) -> None:
        self.seed = seed
        self.rng = default_rng(seed)
        self.max_nodes = max_nodes
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger("generator")


class GraphGeneratorAlt(BaseGenerator):
    def __init__(self, seed, max_nodes):
        super().__init__(seed, max_nodes)
        self.name_generator = NameGenerator("kyoiku-kanji-2017.csv")
        self.total_nodes = 1
        self.max_children = 3
        self.max_ttc = 100

    def generate_graph(self):

        i = 0
        for _ in range(self.max_nodes):
            step_type = self.rng.choice([AND, OR])
            self.graph.add_node(i, step_type=step_type)
            i += 1

        unvisited = list(self.graph.nodes)
        visited = set()
        while len(unvisited) > 1:
            current_node = self.rng.choice(unvisited)

            # Add children
            num_children = self.rng.integers(1, self.max_children)
            if num_children != 0:
                unvisited.remove(current_node)
                visited.add(current_node)
                children = []
                for n in range(num_children):
                    child = int(self.rng.choice(unvisited))
                    child_type = self.graph.nodes[child]["step_type"]
                    self.graph.add_edge(current_node, child, step_type=child_type)
                    children.append(child)

            # Add parents
            num_parents = self.rng.integers(self.max_children)
            # successors = get_all_successors(self.graph, current_node)
            predecessors = list(set(visited))
            if len(predecessors) > 0 and num_parents != 0:
                for _ in range(num_parents):
                    parent = int(self.rng.choice(predecessors))
                    if not self.graph.has_edge(parent, current_node):
                        self.graph.add_edge(
                            parent,
                            current_node,
                            step_type=self.graph.nodes[current_node]["step_type"],
                        )

            current_node = random.choice(list(children))

            pass

        return self.graph


def get_all_successors(graph: nx.DiGraph, node, children=None, visited=None):

    if children is None:
        children = set()
    else:
        children.add(node)

    if graph.out_degree(node) == 0:
        return {}
    else:
        for child in graph.successors(node):
            children.union(get_all_successors(graph, child, children, visited))

        return children


class GraphGenerator(BaseGenerator):
    def __init__(self, seed, max_nodes):
        super().__init__(seed, max_nodes)
        self.name_generator = NameGenerator("kyoiku-kanji-2017.csv")
        self.graph.add_node(self.name_generator.next())
        self.total_nodes = 1
        self.max_children = 5
        self.max_ttc = 100

    def generate_graph(self):
        parent = self.name_generator.current
        while self.total_nodes < self.max_nodes:
            self.add_children(parent)

        return self.graph

    def add_children(self, parent):
        num_children = self.rng.integers(self.max_children)

        # Add Children
        for _ in range(num_children):
            if self.total_nodes > self.max_nodes:
                return
            else:
                node_type = self.rng.choice([AND, OR])
                self.graph.add_node(self.name_generator.next(),)
                self.graph.add_edge(
                    parent,
                    self.name_generator.current,
                    ttc=self.rng.integers(1, self.max_ttc),
                )
                self.total_nodes += 1

        # Recurse
        children = list(self.graph.successors(parent))
        random.shuffle(children)
        for child in children:
            self.add_children(child)

    def create_cross_connections(self):
        nodes = list(self.graph.nodes)
        for node in nodes:
            if self.rng.random() > 0.8:
                self.graph.add_edge(node, self.rng.choice(nodes))


class DirectedGraphGenerator(BaseGenerator):
    def __init__(self, seed, max_nodes):
        super().__init__(seed, max_nodes)

    def select_next(self):
        nodes = list(self.graph.nodes)
        return self.rng.choice(nodes)

    def generate_graph(self):

        current_node = 0
        step_type = self.rng.choice([AND, OR])
        self.graph.add_node(current_node, step_type=step_type, asset=0, ttc=10)
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
            if attributes["step_type"] == AND:
                required_steps = self.graph.predecessors(node)
                for step in required_steps:
                    paths = nx.shortest_path(self.graph, source=entry_node, target=step)
                    if len(paths) == 0:
                        unreachable.append(node)
                        valid = False

        return valid, unreachable

    def set_edge_attributes(self):

        for node, attributes in self.graph.nodes().items():
            step_type = attributes["step_type"]
            for predecessor in self.graph.predecessors(node):
                self.graph.add_edge(predecessor, node, step_type=step_type)

    def assign_assets(self, entrypoint):
        """Has to be done before lateral edges are added"""
        children = self.graph.successors(entrypoint)
        self.asset_count = 1 # Start from 1 since root is first asset

        mappings = {}
        for c in children:
            mappings[c] = self.asset_count
            for d in nx.descendants(self.graph, c):
                mappings[d] = self.asset_count
            self.asset_count += 1

        nx.set_node_attributes(self.graph, mappings, name="asset")

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
            if attributes["step_type"] == AND:
                data["step_type"] = "and"
            elif attributes["step_type"] == OR:
                data["step_type"] = "or"

            data["ttc"] = TTC_EASY

            if "reward" in attributes:
                data["reward"] = attributes["reward"]           

            data["children"] = []
            for s in self.graph.successors(n):
                data["children"].append(self.node_to_string(s))

            data["conditions"] = [str(attributes["asset"])]

            to_write[name] = data
            
        with open(f"{filename}", "w") as f:
            yaml.dump(to_write, f)


    def add_more_assets(self):
        # high degree
        # parent is from same asset
        # many descendants with same asset as parent

        def node_has_high_out(node, descendants):
            if self.graph.out_degree(node) > self.rng.integers(1, 5) and self.graph.in_degree(node) == 1:
                self.graph.nodes[node]["asset"] = self.asset_count
                for n in descendants:
                    self.graph.nodes[n]["asset"] = self.asset_count
                self.asset_count += 1

        def node_has_many_children(node, descendants):
            if len(descendants) > self.rng.integers(2, 3):
                self.graph.nodes[node]["asset"] = self.asset_count
                for n in descendants:
                    self.graph.nodes[n]["asset"] = self.asset_count
                self.asset_count += 1

        conditions = [node_has_high_out, node_has_many_children]
        for func in conditions:
            for node, attributes in self.graph.nodes.items():
                
                try:
                    parent = self.graph.successors(node).__next__()
                except StopIteration:
                    continue
                parent_asset = self.graph.nodes[parent]["asset"]
                descendants = nx.descendants(self.graph, node)
                asset = attributes["asset"]
                if asset == parent_asset:
                    func(node, descendants)
                    pass



def add_unique(path: list, item):
    if item not in path:
        path.append(item)


class PathFinderAttacker():

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

        path: list = nx.shortest_path(self.attack_graph, source=self.start_node, target=target, weight="ttc")

        # Check each step in the path.
        for node_id in path:
            step = self.attack_graph.nodes[node_id]
            # If node is AND step, go to parents first.
            if step["step_type"] == AND and node_id not in self.total_path:
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


def generate_graph(name, size, lateral_connections, num_flags):
    pass



def main(seed=888):

    # seed=None


    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger("main")

    random.seed(seed)

    initial_step = 0
    size = 200
    lateral_connections = 50
    num_flags = 20
    graph_name="big"

    graph_generator = DirectedGraphGenerator(seed, size)

    attack_graph = graph_generator.generate_graph()

    graph_generator.assign_assets(initial_step)

    # If an AND step only has one parent, consider it as an OR step
    # for node_key in attack_graph.nodes():
    #     node = attack_graph.nodes()[node_key]
    #     if node["step_type"] == AND and len(list(attack_graph.predecessors(node_key))) == 1:
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


    graph_generator.to_yaml(f"{graph_name}.yaml")

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

    #count assets

    assets = set()
    for asset in nx.get_node_attributes(attack_graph, "asset").values():
        assets.add(asset)

    logger.info(f"The graph contains {attack_graph.number_of_nodes()} steps with {len(assets)} assets.")

    logger.info("The graph has %d flags.", num_flags)

    json_data = nx.node_link_data(attack_graph)

    attacker = PathFinderAttacker(attack_graph, start_node=0)

    #path = attacker.find_path_to(72)

    #print(path)

    draw_graph(attack_graph, seed, f"{graph_name}.pdf")

    with open(f"{graph_name}.json", "w") as f:
        json.dump(nx.node_link_data(attack_graph), f)

    pass


if __name__ == "__main__":
    main()

