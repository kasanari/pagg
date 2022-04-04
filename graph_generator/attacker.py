import networkx as nx
from .graph import AND, STEP_TYPE
from operator import itemgetter

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