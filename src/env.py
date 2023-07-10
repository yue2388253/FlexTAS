from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
import logging
import numpy as np
import networkx as nx
from typing import SupportsFloat, Any, Optional

from src.network.net import Duration, Flow, Link, transform_line_graph, Net
from src.lib.operation import Operation, check_operation_isolation


class NetEnv(gym.Env):
    def __init__(self, graph: nx.Graph, flows: list[Flow]):
        super().__init__()

        self.graph: nx.Graph = graph
        self.flows: list[Flow] = flows
        self.num_flows: int = len(flows)
        self.line_graph: nx.Graph
        self.link_dict: dict[str, Link]
        self.line_graph, self.link_dict = transform_line_graph(graph)

        self.flows_operations: dict[Flow, list[tuple[Link, Operation]]] = defaultdict(list)
        self.links_operations: dict[Link, list[tuple[Flow, Operation]]] = defaultdict(list)

        self.flows_scheduled: list[int] = [0 for _ in range(self.num_flows)]

        self.last_action = None

        self.reward: float = 0

        self.state: ObsType = None

        self.observation_space = spaces.Dict({
            'adjacency_matrix': spaces.MultiBinary([len(self.link_dict), len(self.link_dict)]),
            'nodes_features': spaces.Box(low=0, high=1, shape=(len(self.link_dict), 3))
        })

        # action space:
        # 1. which flow to schedule.
        # 2. enable gating or not.
        self.action_space = spaces.MultiDiscrete([len(self.flows), 2])

        self.reset()

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        super().reset(seed=seed)

        self.flows_operations.clear()
        self.links_operations.clear()

        self.flows_scheduled = [0 for _ in range(self.num_flows)]

        self.reward = 0

        self.state = self._generate_state()

        return self.state, {}

    def _generate_state(self) -> ObsType:
        # todo: generate state
        adjacency_matrix = nx.to_scipy_sparse_array(self.line_graph).todense().astype(np.int8)
        features = np.random.uniform(size=(len(self.link_dict), 3)).astype(np.float32)
        return {
            'adjacency_matrix': adjacency_matrix,
            'nodes_features': features
        }

    def get_action_mask(self) -> list[int]:
        return self.flows_scheduled

    # return None if valid, else return the conflict operation
    def check_valid_flow(self, flow: Flow) -> Optional[int]:
        for link, operation in self.flows_operations[flow]:
            offset = self.check_valid_link(link)
            if isinstance(offset, int):
                return offset
        return None

    def check_valid_link(self, link: Link) -> Optional[int]:
        # only needs to check whether the newly added operation is conflict with other operations.
        flow, operation = self.links_operations[link][-1]
        safe_distance = link.safe_distance()

        for flow_rhs, operation_rhs in self.links_operations[link][:-1]:
            offset = check_operation_isolation(
                (operation, flow.period),
                (operation_rhs, flow_rhs.period),
                safe_distance
            )
            if offset is not None:
                return offset
        return None

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        flow_index, gating = action

        flow = self.flows[flow_index]
        gating = (gating == 1)

        self.last_action = (flow_index, gating)

        assert self.flows_scheduled[flow_index] == 0
        hop_index = len(self.flows_operations[flow])

        link = self.link_dict[flow.path[hop_index]]

        # change gating to False in case it is invalid to enable gating due to GCL constraint.
        # todo: this might make learning hard, since the agent does not know the gating action is changed.
        if gating:
            try:
                link.add_gating(flow.period)
            except RuntimeError:
                gating = False

        if hop_index == 0:
            operation = Operation(0, 0 if gating else None, 0, link.transmission_time(flow.payload))
        else:
            last_link, last_operation = self.flows_operations[flow][hop_index - 1]
            last_link_earliest, last_link_latest = last_operation.earliest_time, last_operation.latest_time
            last_link_earliest + last_link.transmission_time(flow.payload) + Net.DELAY_PROP + Net.DELAY_PROC_MIN,
            earliest_time = last_link_earliest + last_link.transmission_time(
                flow.payload) + Net.DELAY_PROP + Net.DELAY_PROC_MIN
            latest_time = last_link_latest + last_link.transmission_time(
                flow.payload) + Net.SYNC_PRECISION + Net.DELAY_PROP + Net.DELAY_PROC_MAX
            gating_time = latest_time if gating else None
            end_time = latest_time + link.transmission_time(flow.payload)

            operation = Operation(earliest_time, gating_time, latest_time, end_time)

        self.flows_operations[flow].append((link, operation))
        self.links_operations[link].append((flow, operation))

        scheduled = False
        while True:
            offset = self.check_valid_flow(flow)
            if offset is None:
                scheduled = True
                break

            assert isinstance(offset, int)

            for link, operation in self.flows_operations[flow]:
                operation.add(offset)
                if operation.end_time > flow.period:
                    # cannot be scheduled
                    break

        done = False
        if scheduled:
            self.reward += 0.1

            if len(flow.path) == len(self.flows_operations[flow]):
                self.flows_scheduled[flow_index] = 1
                done = all(self.flows_scheduled)
        else:
            done = True
            self.reward -= 100

        self.state = self._generate_state()

        self.render()
        return self.state, self.reward, done, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        logging.debug(self.flows_scheduled)
        logging.debug(self.last_action)

        flow_index, gating = self.last_action

        logging.debug(self.flows_operations[self.flows[flow_index]][-1])
        return

    def close(self):
        return
