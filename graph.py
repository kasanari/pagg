from attr import attributes
import networkx as nx
import matplotlib.pyplot as plt

import random

from numpy.random import default_rng
import numpy as np

import logging

import json

AND = "AND"
OR = "OR"
DEFENSE = "DEFENSE"

ASSET = "asset"
STEP_TYPE = "step_type"
CONDITIONS = "conditions"

TTC_EASY = "EASY_TTC"
TTC_HARD = "HARD_TTC"

REWARD_EASY = "LOW_FLAG_REWARD"
REWARD_MEDIUM = "MEDIUM_FLAG_REWARD"
REWARD_HARD = "HIGH_FLAG_REWARD"


def draw_graph(graph: nx.DiGraph, seed, outfile=None, num_assets=None):

    lengths = list(nx.shortest_path_length(graph, source=0).values())
    node_colors = np.array(list(nx.get_node_attributes(graph, ASSET).values()))

    # if num_assets is not None:
    #    node_colors = node_colors / num_assets

    node_size = 300
    node_options = {
        "node_size": node_size,
        #"node_color": node_colors,
        "edgecolors": "black",
        "linewidths": 2,
    }

    defense_options = {
        "node_size": node_size,
        "edgecolors": "black",
        "linewidths": 2,
    }

    edge_options = {"width": 2, "node_size": node_size}

    label_options = {
        "font_size": 10,
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

    attack_steps = [step for step, attributes in graph.nodes.items() if attributes[STEP_TYPE] != DEFENSE]
    defense_steps = [step for step, attributes in graph.nodes.items() if attributes[STEP_TYPE] == DEFENSE]

    plt.figure(figsize=(16, 9))
    nx.draw_networkx_edges(graph, pos, edgelist=OR_edges, **edge_options)
    nx.draw_networkx_edges(graph, pos, edgelist=AND_edges, style="dashed", **edge_options)
    nx.draw_networkx_nodes(
        graph, pos, nodelist=attack_steps, **node_options, cmap=plt.cm.get_cmap("tab20")
    )  # default spring_layout
    nx.draw_networkx_nodes(
        graph, pos, nodelist=defense_steps, **defense_options, cmap=plt.cm.get_cmap("tab20")
    )
    nx.draw_networkx_labels(graph, pos, **label_options)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    # ax.margins(0.20)
    plt.axis("off")
    plt.tight_layout()

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)

    plt.close()


def draw_instance_model(graph, filename):
    node_size = 600

    node_colors = np.array(graph.nodes)

    # node_colors = node_colors / len(node_colors)

    options = {
        "node_size": node_size,
        "edgecolors": "black",
        "node_color": node_colors,
        "linewidths": 2,
        "width": 2,
        "font_size": 16,
        "font_color": "black",
    }
    plt.figure(figsize=(4.33, 2.66))
    pos = nx.nx_pydot.graphviz_layout(graph, prog="dot")

    nx.draw_networkx(graph, pos, **options, cmap=plt.cm.get_cmap("tab20"))
    ax = plt.gca()
    # ax.margins(0.20)
    plt.axis("off")
    plt.tight_layout()
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
        self.graph: nx.DiGraph = nx.DiGraph()
        self.logger = logging.getLogger("generator")


class DirectedGraphGenerator(BaseGenerator):
    def __init__(self, seed, max_nodes):
        super().__init__(seed, max_nodes)
        self.instance_model = nx.DiGraph()
        self.asset_count = 0
        self.step_count = 0

    def select_next(self):
        nodes = list(self.graph.nodes)
        return self.rng.choice(nodes)

    def generate_initial_graph(self):
        # Create root step
        self.create_new_step(step_type=self.rng.choice([AND, OR]), asset=self.create_new_asset(obj_class="internet", unmalleable=True))
        for _ in range(self.asset_count, self.max_nodes):
            step_type = self.rng.choice([AND, OR])
            # Select next node to grow from
            parent = self.select_next()
            # Create the step
            self.create_new_step(step_type, parent=parent)

        return self.graph

    def create_lateral_connections(self, num_connections):

        all_nodes = set(list(self.graph.nodes()))
        to_connect = list(self.graph.nodes())
        to_connect.remove(0)
        new_connections = []
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
                new_connections.append((new_child, node))

            pass

        # Add the new edges
        self.graph.add_edges_from(new_connections)
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
            new_asset = self.create_new_asset(obj_class="server")
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
        new_assets = {}
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
            new_assets[f] = self.create_new_asset(
                self.get_asset(f), flag=True, unmalleable=True
            )
            if d <= medium_bar:
                flags_with_rewards[f] = REWARD_EASY
            elif d <= hard_bar:
                flags_with_rewards[f] = REWARD_MEDIUM
            else:
                flags_with_rewards[f] = REWARD_HARD

        nx.set_node_attributes(self.graph, new_assets, ASSET)
        nx.set_node_attributes(self.graph, flags_with_rewards, "reward")
        return len(flags)

    def node_to_string(self, node):
        asset_str = self.asset_to_string(self.get_asset(node))

        if "reward" in self.graph.nodes[node]:
            name = f"{asset_str}:take"
        else:
            name = f"{asset_str}:{node}"

        return name

    def get_all_attack_steps(self, asset):
        attack_steps = {
            node
            for node, attributes in self.graph.nodes.items()
            if attributes["asset"] == asset
        }
        for descendant in nx.descendants(self.instance_model, asset):
            attack_steps |= self.get_all_attack_steps(descendant)
        return attack_steps

    def attach_defense_step_to_graph(self, attack_steps, defense_asset):
        attached_steps = []
        defense_step = self.create_new_step(step_type=DEFENSE, asset=defense_asset)
        for step in attack_steps:
            if self.graph.nodes[step][STEP_TYPE] == OR:
                new_AND_step = self.create_new_step(step_type=AND, parent=defense_step, asset=defense_asset)
                new_OR_step = self.create_new_step(step_type=OR, asset=defense_asset)
                # Connect OR-step and step to AND-step
                self.graph.add_edges_from(
                    [(new_AND_step, step), (new_OR_step, new_AND_step)]
                )
                # Connect parents to OR-step
                parents = list(self.graph.predecessors(step)).copy()
                for p in parents:
                    self.graph.add_edge(p, new_OR_step)
                    self.graph.remove_edge(p, step)
                attached_steps.append(new_AND_step)
            elif self.graph.nodes[step][STEP_TYPE] == AND:
                self.graph.add_edge(defense_step, step)
                attached_steps.append(step)
            else:
                continue
        
        

    def calculate_defense_cost(self, asset):
        return len(nx.descendants(self.instance_model, asset)) + 1

    def generate_defense_steps(self):
        for asset, attributes in self.instance_model.nodes.items():
            if attributes["unmalleable"]:
                continue
            attack_steps = self.get_all_attack_steps(asset)
            self.attach_defense_step_to_graph(attack_steps, asset)

    def get_all_defense_steps(self):
        return [key for key, value in self.graph.nodes.items() if value[STEP_TYPE] == DEFENSE]

    def serialize_defense_steps(self):
        defense_steps = []
        for defense in self.get_all_defense_steps():
            attributes = self.graph.nodes[defense]
            asset = attributes["asset"]
            defense_step = {STEP_TYPE: DEFENSE}
            #defense_step["objclass"] = attributes["obj_class"]
            defense_step["reward"] = self.calculate_defense_cost(asset)
            defense_step["atkname"] = "defend"
            defense_step["assets_disabled"] = [
                self.asset_to_string(descendant)
                for descendant in nx.descendants(self.instance_model, asset)
            ]
            defense_step["id"] = ":".join(
                [self.asset_to_string(asset), defense_step["atkname"]]
            )
            defense_step["links"] = sorted(self.graph.successors(defense))

            defense_steps.append(defense_step)

        return defense_steps

    def get_asset_type(self, asset):
        is_flag = self.instance_model.nodes[asset]["flag"]
        if is_flag:
            obj_type = "flag"
        else:
            obj_type = "asset"
        return obj_type

    def asset_to_string(self, asset):
        obj_type = self.get_asset_type(asset)
        return f"{obj_type}:{asset}"

    def to_file(self, filename):
        attack_graph = []
        for n, attributes in self.graph.nodes.items():
            name = self.node_to_string(n)
            data = {}

            if attributes[STEP_TYPE] == AND:
                data[STEP_TYPE] = "and"
            elif attributes[STEP_TYPE] == OR:
                data[STEP_TYPE] = "or"

            data["ttc"] = TTC_EASY

            data["reward"] = attributes["reward"] if "reward" in attributes else None

            data["links"] = [self.node_to_string(s) for s in self.graph.successors(n)]
            data["objclass"] = self.get_asset_type(self.get_asset(n))
            data["id"] = name
            data["atkname"] = n

            attack_graph.append(data)

        defense_steps = self.serialize_defense_steps()

        attack_graph.extend(defense_steps)

        with open(f"{filename}", "w") as f:
            json.dump(attack_graph, f, indent=" ")

    def create_new_step(self, step_type, parent=None, asset=None, ttc=10):
        new_step = self.step_count
        self.graph.add_node(new_step, step_type=step_type, ttc=ttc, asset=asset)
        if parent is not None:
            self.graph.add_edge(parent, new_step, ttc=ttc)
        self.step_count += 1
        return new_step

    def create_new_asset(
        self, parent=None, flag=False, obj_class=None, unmalleable=False
    ):
        new_asset = self.asset_count
        self.instance_model.add_node(
            new_asset, flag=flag, obj_class=obj_class, unmalleable=unmalleable
        )
        if parent is not None:
            self.instance_model.add_edge(parent, new_asset)
        self.asset_count += 1
        return new_asset

    def get_asset(self, node):
        return self.graph.nodes[node][ASSET]

    def add_more_assets(self):
        # high degree
        # parent is from same asset
        # many descendants with same asset as parent

        def update_node(node, asset):
            # Set the asset of the node
            old_asset = self.get_asset(node)
            self.graph.nodes[node][ASSET] = asset
            # Add the old asset as a condition
            self.graph.nodes[node][CONDITIONS].add(old_asset)
            self.graph.nodes[node][CONDITIONS].add(asset)

        def node_has_high_out(node, descendants):
            if self.graph.out_degree(node) > 5 and self.graph.in_degree(node) == 1:
                old_asset = self.get_asset(node)
                new_asset = self.create_new_asset(old_asset, obj_class="host")
                update_node(node, new_asset)
                for n in descendants:
                    update_node(n, new_asset)

        def node_has_many_children(node, descendants):
            if len(descendants) > 5:
                old_asset = self.get_asset(node)
                new_asset = self.create_new_asset(old_asset, obj_class="host")
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
                parent_asset = self.get_asset(parent)
                descendants = nx.descendants(self.graph, node)
                asset = attributes[ASSET]
                if asset == parent_asset:
                    func(node, descendants)
                    pass




def generate_graph(name, size, lateral_connections, num_flags, seed):

    log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger("main")

    random.seed(seed)

    initial_step = 0

    graph_generator = DirectedGraphGenerator(seed, size)

    attack_graph = graph_generator.generate_initial_graph()

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

    graph_generator.generate_defense_steps()

    cc = nx.clustering(attack_graph)
    avg_cc = np.mean(list(cc.values()))
    logger.info(f"Average CC: {avg_cc}")

    in_degress = attack_graph.in_degree()
    out_degrees = attack_graph.out_degree()

    graph_generator.to_file(f"model_{name}.json")

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

    draw_graph(
        attack_graph,
        seed,
        f"ag_{name}.pdf",
        num_assets=graph_generator.instance_model.number_of_nodes(),
    )
    draw_instance_model(graph_generator.instance_model, f"instance_{name}.pdf")

    conditions = nx.get_node_attributes(attack_graph, CONDITIONS)
    conditions = {k: list(v) for k, v in conditions.items()}
    nx.set_node_attributes(attack_graph, conditions, CONDITIONS)

    with open(f"nx_{name}.json", "w") as f:
        json.dump(nx.node_link_data(attack_graph), f)

    pass


def main(seed=888):

    configs = [
        dict(name="small", size=50, lateral_connections=5, num_flags=5,),
    #    dict(name="medium", size=100, lateral_connections=25, num_flags=10,),
    #    dict(name="big", size=200, lateral_connections=50, num_flags=20,),
    ]

    for config in configs:
        generate_graph(**config, seed=seed)


if __name__ == "__main__":
    main()

