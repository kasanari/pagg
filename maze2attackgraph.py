import json
import networkx as nx

from graph_generator.graph import AttackGraph, InstanceModel
import graph_generator.graph_utils as graph_utils
from graph_generator.drawing import draw_attack_graph

FLOOR = 0
ATTACKER = 2
DOOR = 3
KEY = 4
FLAG = 5

KEY_GET = 6
DOOR_REACH = 7
DEFENSE = 8
DEFENSE_ENABLE = 9

"""
 {
  "step_type": "or",
  "ttc": "EASY_TTC",
  "reward": null,
  "links": [],
  "objclass": "asset",
  "id": "asset:47:180",
  "atkname": 180
 },
"""

tile2attackname = {
    FLOOR: "connect",
    ATTACKER: "enter",
    KEY: "gather",
    FLAG: "gather",
    KEY_GET: "credentials_gathered",
    DOOR: "unlock",
    DOOR_REACH: "discover",
    DEFENSE: "connect",
    DEFENSE_ENABLE: "enable",
}

tile2asset = {
    FLOOR: "host",
    ATTACKER: "attacker",
    DOOR: "authenticator",
    DOOR_REACH: "authenticator",
    KEY: "credentials",
    KEY_GET: "credentials",
    FLAG: "flag",
    DEFENSE: "firewall",
}

graph_name = "4ways"

with open(f"{graph_name}.json", encoding="utf8") as f:
    maze_graph = json.load(f)

digraph: nx.DiGraph = nx.node_link_graph(maze_graph, directed=True, multigraph=False)

attack_graph = AttackGraph()
instance_model = InstanceModel()

for node, d in digraph.nodes(data=True):
    is_flag = d["tile"] == FLAG
    unmalleable = d["tile"] == ATTACKER

    if d["tile"] in (KEY_GET, DOOR_REACH, DEFENSE_ENABLE):
        asset_id = d["asset_id"]
    else:
        asset_id = node
        instance_model.graph.add_node(
            asset_id, obj_class=tile2asset[d["tile"]], is_flag=is_flag, unmalleable=unmalleable
        )

    if d["tile"] == FLAG:
        reward = 3
    elif d["tile"] == DEFENSE_ENABLE:
        reward = "defense_default"
    else:
        reward = 0

    attack_graph.graph.add_node(
        node,
        step_type=d["step_type"],
        step_name=tile2attackname[d["tile"]],
        ttc=d["ttc"],
        reward=reward,
        asset=asset_id,
    )

for u, v, d in digraph.edges(data=True):
    attack_graph.graph.add_edge(u, v)

for step in attack_graph.defense_steps:
    descendants = nx.descendants(attack_graph.graph, step.id)
    for d in descendants:
        attack_graph.graph.add_edge(step.id, d)


draw_attack_graph(attack_graph, None)

with open(f"{graph_name}.yaml", "w", encoding="utf8") as f:
    graph_utils.save_to_file(attack_graph, instance_model, f)
