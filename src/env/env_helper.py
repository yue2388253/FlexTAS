import os
from src.env.env import NetEnv
from src.network.net import generate_linear_5, generate_flows
from src.network.from_json import generate_net_flows_from_json


def linear_5(num_flows: int, seed: int = None) -> NetEnv:
    graph = generate_linear_5()
    flows = generate_flows(graph, num_flows, seed=seed)
    env = NetEnv(graph, flows)
    return env


def from_file(filename: os.PathLike) -> NetEnv:
    assert os.path.isfile(filename), f"No such file {filename}"
    graph, flows = generate_net_flows_from_json(filename)
    return NetEnv(graph, flows)
