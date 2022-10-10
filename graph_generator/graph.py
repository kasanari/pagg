from typing import Union
from dataclasses import dataclass, field
from typing import List, Set

import networkx as nx

from . import nx_utils
from .constants import ASSET, CONDITIONS, TTC, STEP


@dataclass
class AttackStep:
    id: int
    step_type: str
    asset: int
    ttc: Union[int, TTC] = TTC.DEFAULT
    assets_disabled: List[int] = field(default_factory=list)
    conditions: Set[int] = field(default_factory=set)
    step_name: str = ""


@dataclass
class Asset:
    id: int
    is_flag: bool
    unmalleable: bool
    obj_class: str = "asset"


class InstanceModel:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.asset_count = 0

    def add_asset(self, parent=None, flag=False, obj_class=None, unmalleable=False):
        new_asset = self.asset_count
        self.graph.add_node(new_asset, is_flag=flag, obj_class=obj_class, unmalleable=unmalleable)
        if parent is not None:
            self.graph.add_edge(parent, new_asset)
        self.asset_count += 1
        return new_asset

    def __getitem__(self, node):
        return Asset(id=node, **self.graph.nodes[node])

    def __iter__(self):
        return (Asset(id=key, **values) for key, values in self.graph.nodes.items())

    def calculate_defense_cost(self, asset):
        return len(nx.descendants(self.graph, asset)) + 1

    def get_asset_type(self, asset):
        if self[asset].is_flag:
            return "flag"
        
        return self[asset].obj_class


    def asset_to_string(self, asset):
        obj_type = self.get_asset_type(asset)
        return f"{obj_type}:{asset}"


class AttackGraph:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.step_count = 0
        self.entrypoint = 0

    def add_step(self, step_type: STEP, parent=None, asset=None, ttc=TTC.DEFAULT, assets_disabled=None, name=""):
        new_step = self.step_count
        self.graph.add_node(
            new_step, step_type=step_type, ttc=ttc, asset=asset, assets_disabled=assets_disabled, step_name=name
        )
        if parent is not None:
            self.graph.add_edge(parent, new_step, ttc=ttc)
        self.step_count += 1
        return new_step

    def __iter__(self):
        return (AttackStep(id=key, **values) for key, values in self.graph.nodes.items())

    def __getitem__(self, node):
        return AttackStep(id=node, **self.graph.nodes[node])

    def children(self, node):
        return self.graph.successors(node)

    @property
    def defense_steps(self):
        return filter(lambda step: step.step_type == STEP.DEFENSE, self)

    @property
    def attack_steps(self):
        return filter(lambda step: step.step_type != STEP.DEFENSE, self)

    @property
    def steps(self):
        return sorted(self.graph.nodes)

    def check_AND_reachability(self):
        return nx_utils.check_AND_reachability(self.graph, self.entrypoint)

    def update_step(self, node: int, new_asset: int):
        # Set the asset of the node
        old_asset = self[node].asset
        self.graph.nodes[node][ASSET] = new_asset
        # Add the old asset as a condition
        self.graph.nodes[node][CONDITIONS].add(old_asset)
        self.graph.nodes[node][CONDITIONS].add(new_asset)
        return (node, old_asset)
