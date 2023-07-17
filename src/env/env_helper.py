import os
from src.env.env import NetEnv
from src.network.net import generate_linear_5, generate_cev, generate_flows
from src.network.from_json import generate_net_flows_from_json


def generate_env(topo: str = "L5", num_flows: int = 10, seed: int = None) -> NetEnv:
    if topo == "L5":
        graph = generate_linear_5()
    elif topo == "CEV":
        graph = generate_cev()
    else:
        raise ValueError(f"Unknown topo type.")

    flows = generate_flows(graph, num_flows, seed=seed)

    env = NetEnv(graph, flows)
    return env


def from_file(filename: os.PathLike) -> NetEnv:
    assert os.path.isfile(filename), f"No such file {filename}"
    graph, flows = generate_net_flows_from_json(filename)
    return NetEnv(graph, flows)
