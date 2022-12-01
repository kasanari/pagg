from itertools import chain
from typing import Dict
import matplotlib.pyplot as plt
import networkx as nx

from graph_generator.graph import AttackGraph

from .constants import STEP_TYPE, TTC, REWARD, STEP

ttc_labels = {
    TTC.NONE: "",
    TTC.DEFAULT: "d",
    TTC.EASY: "e",
    TTC.MEDIUM: "m",
    TTC.HARD: "h",
}

reward_labels = {
    REWARD.DEFAULT: "D",
    REWARD.EASY: "E",
    REWARD.MEDIUM: "M",
    REWARD.HARD: "H",
}


def draw_attack_graph(attack_graph: AttackGraph, flags: Dict[str, REWARD], _, outfile=None):

    graph: nx.DiGraph = attack_graph.graph
    # lengths = list(nx.shortest_path_length(graph, source=0).values())
    attack_node_colors_dict = {step.id: "#ffffff" for step in attack_graph.attack_steps if step.id not in flags}
    attack_node_colors_dict[0] = "#ff0000"

    children_of_defense_steps = chain(
        *[list(attack_graph.graph.successors(step.id)) for step in attack_graph.defense_steps]
    )
    for child in children_of_defense_steps:
        attack_node_colors_dict[child] = "#800080"

    attack_node_colors = [
        attack_node_colors_dict[step.id] for step in attack_graph.attack_steps if step.id not in flags
    ]

    # if num_assets is not None:
    #    node_colors = node_colors / num_assets

    node_size = 200
    node_options = {
        "node_size": node_size,
        "node_color": attack_node_colors,
        "edgecolors": "black",
        "linewidths": 1,
    }

    defense_options = {
        "node_size": node_size,
        "node_color": "#cbc3e3",
        "edgecolors": "black",
        "linewidths": 1,
    }

    flag_options = {
        "node_size": node_size,
        "node_color": "#00ff00",
        "edgecolors": "black",
        "linewidths": 1,
    }

    edge_options = {"width": 1, "node_size": node_size}

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

        if child_type == STEP.OR:
            OR_edges.append((p, c))
        elif child_type == STEP.AND:
            AND_edges.append((p, c))
        else:
            raise Exception(f"Invalid child type {child_type}")

    attack_steps = {step.id for step in attack_graph.attack_steps}
    defense_steps = {step.id for step in attack_graph.defense_steps}

    flag_steps = set(flags.keys())
    attack_steps -= flag_steps

    plt.figure(figsize=(5, 5))
    nx.draw_networkx_edges(graph, pos, edgelist=OR_edges, **edge_options)
    nx.draw_networkx_edges(graph, pos, edgelist=AND_edges, style="dashed", **edge_options)
    nx.draw_networkx_nodes(
        graph, pos, nodelist=attack_steps, **node_options, cmap=plt.cm.get_cmap("tab20")
    )  # default spring_layout
    nx.draw_networkx_nodes(graph, pos, nodelist=defense_steps, **defense_options)
    nx.draw_networkx_nodes(graph, pos, nodelist=flag_steps, **flag_options)

    step_labels = {step.id: ttc_labels.get(step.ttc, step.ttc) for step in attack_graph.attack_steps}
    root_label = {0: "A"}
    flag_labels = {n: "F" for n, r in flags.items()}  # {n:reward_labels[r] for n, r in flags.items()}
    defense_labels = {n: "D" for n in defense_steps}
    step_labels |= root_label | flag_labels | defense_labels
    nx.draw_networkx_labels(graph, pos, labels=step_labels, **label_options)
    # nx.draw_networkx_labels(graph, pos, labels = defense_labels, **label_options)
    # nx.draw_networkx_labels(graph, pos, labels = root_label, **label_options)
    # nx.draw_networkx_labels(graph, pos, labels =  , **label_options)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0)
    plt.axis("off")
    plt.tight_layout()

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)

    plt.close()


def draw_instance_model(graph, filename):
    node_size = 600

    node_colors = list(graph.nodes)

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
    ax.margins(0.20)
    plt.axis("off")
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    plt.close()
