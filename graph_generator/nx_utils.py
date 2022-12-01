import networkx as nx
from .constants import STEP, STEP_TYPE
from itertools import filterfalse
from networkx.algorithms.shortest_paths.generic import shortest_path


def validate_graph(graph: nx.DiGraph, entrypoint):
    reachable, _ = check_AND_reachability(graph, entrypoint)

    acyclic = nx.is_directed_acyclic_graph(graph)

    return reachable, acyclic


def check_AND_reachability(graph: nx.DiGraph, entry_node):
    def step_is_reachable(node):
        try:
            shortest_path(graph, source=entry_node, target=node)
        except nx.NetworkXNoPath:
            return False
        return True

    def conditions_reachable(node):
        return (
            step_is_reachable(parent)
            for parent in graph.predecessors(node)
            if graph.nodes[parent][STEP_TYPE] != STEP.DEFENSE
        )

    unreachable = filterfalse(lambda node: all(conditions_reachable(node)), graph.nodes)

    return len(list(unreachable)) == 0, unreachable
