import matplotlib.pyplot as plt
import networkx as nx

from graph_generator.graph import AttackGraph

from .constants import AND, ASSET, DEFENSE, OR, STEP_TYPE


def draw_attack_graph(attack_graph: AttackGraph, _, outfile=None):

    graph: nx.DiGraph = attack_graph.graph
    #lengths = list(nx.shortest_path_length(graph, source=0).values())
    attack_node_colors = [step.asset for step in attack_graph.attack_steps if not step.is_flag]

    # if num_assets is not None:
    #    node_colors = node_colors / num_assets

    node_size = 300
    node_options = {
        "node_size": node_size,
        "node_color": attack_node_colors,
        "edgecolors": "black",
        "linewidths": 2,
    }

    defense_options = {
        "node_size": node_size,
        "node_color": "blue",
        "edgecolors": "black",
        "linewidths": 2,
    }

    flag_options = {
        "node_size": node_size,
        "node_color": "yellow",
        "edgecolors": "black",
        "linewidths": 2,
    }

    edge_options = {"width": 2, "node_size": node_size}

    label_options = {
        "font_size": 10,
    }

    # pos = nx.kamada_kawai_layout(graph)
    #pos = nx.spring_layout(graph, k=10, iterations=1000, fixed=[0], pos={0:(0,0)}, center=(0,0), seed=seed)
    # pos = nx.planar_layout(graph, scale=10)
    # pos = nx.spectral_layout(graph)
    pos = nx.nx_pydot.graphviz_layout(graph, root=0, prog="dot")

    OR_edges = []
    AND_edges = []
    for p, c in graph.edges:
        child_type = graph.nodes[c][STEP_TYPE]

        if child_type == OR:
            OR_edges.append((p, c))
        elif child_type == AND:
            AND_edges.append((p, c))
        else:
            raise Exception(f"Invalid child type {child_type}")

    attack_steps = {step.id for step in attack_graph.attack_steps}
    defense_steps = {step.id for step in attack_graph.defense_steps}
    flag_steps = {step for step, attributes in graph.nodes.items() if attributes["reward"] != 0 and attributes["step_type"] != DEFENSE}

    attack_steps -= flag_steps

    plt.figure(figsize=(5, 5))
    nx.draw_networkx_edges(graph, pos, edgelist=OR_edges, **edge_options)
    nx.draw_networkx_edges(graph, pos, edgelist=AND_edges, style="dashed", **edge_options)
    nx.draw_networkx_nodes(
        graph, pos, nodelist=attack_steps, **node_options, cmap=plt.cm.get_cmap("tab20")
    )  # default spring_layout
    nx.draw_networkx_nodes(graph, pos, nodelist=defense_steps, **defense_options)
    nx.draw_networkx_nodes(graph, pos, nodelist=flag_steps, **flag_options)
    
    step_labels = {step.id:step.asset for step in attack_graph.attack_steps}
    root_label = {0: "A"}
    flag_labels = {n:"F" for n in flag_steps}
    defense_labels = {n:"D" for n in defense_steps}
    step_labels |= root_label | flag_labels | defense_labels
    nx.draw_networkx_labels(graph, pos, labels = step_labels, **label_options)
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
