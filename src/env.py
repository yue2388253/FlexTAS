from collections import defaultdict
from enum import Enum, auto
import gymnasium as gym
import pandas as pd
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
import logging
import numpy as np
import networkx as nx
from typing import SupportsFloat, Any, Optional

from src.network.net import Duration, Flow, Link, transform_line_graph, Net
from src.lib.operation import Operation, check_operation_isolation


class ErrorType(Enum):
    AlreadyScheduled = auto()
    JitterExceed = auto()
    PeriodExceed = auto()
    GatingExceed = auto()


class SchedulingError(Exception):
    def __init__(self, error_type: ErrorType, msg):
        super().__init__(f"SchedulingError: {msg}")
        self.error_type: ErrorType = error_type

        
class _StateEncoder:
    def __init__(self, env: 'NetEnv'):
        self.env = env
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, ))

    def state(self):
        # links_id = self.link_dict.keys()
        # one_hot_dict = pd.get_dummies(links_id)
        #
        # # the shape would be (num_operations + 2, num_operations + 2)
        # adjacency_matrx = []
        #
        # # the shape would be (num_operations, num_links) after encoding
        # feature_matrix = []
        #
        # for flow in self.flows:
        #     operations = self.flows_operations[flow]
        #     for hop, link in enumerate(flow.path):
        #         feature_matrix.append(np.array(one_hot_dict[link].values, dtype=np.int8))
        #
        # feature_matrix = np.array(feature_matrix)
        #
        return self.observation_space.sample()


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

        self.state_encoder: _StateEncoder = _StateEncoder(self)

        # self.observation_space = spaces.Dict({
        #     'adjacency_matrix': spaces.MultiBinary([len(self.link_dict), len(self.link_dict)]),
        #     'nodes_features': spaces.Box(low=0, high=1, shape=(len(self.link_dict), 3))
        # })
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, ))

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

        for link in self.link_dict.values():
            link.reset()

        self.flows_operations.clear()
        self.links_operations.clear()

        self.flows_scheduled = [0 for _ in range(self.num_flows)]

        self.reward = 0

        return self._generate_state(), {}

    def _generate_state(self) -> ObsType:
        # todo: generate state
        # adjacency_matrix = nx.to_scipy_sparse_array(self.line_graph).todense().astype(np.int8)
        # features = np.random.uniform(size=(len(self.link_dict), 3)).astype(np.float32)
        # return {
        #     'adjacency_matrix': adjacency_matrix,
        #     'nodes_features': features
        # }
        return self.state_encoder.state()

    def action_masks(self) -> list[int]:
        return [i == 0 for i in self.flows_scheduled] + [True, True]

    def check_valid_flow(self, flow: Flow) -> Optional[int]:
        """
        :param flow:
        :return: None if valid, else return the conflict operation
        """
        # return
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

        try:
            if self.flows_scheduled[flow_index] != 0:
                raise SchedulingError(ErrorType.AlreadyScheduled,
                                      f"This flow {flow.flow_id} has already been scheduled.")

            hop_index = len(self.flows_operations[flow])

            link = self.link_dict[flow.path[hop_index]]

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

                if (not gating) and (hop_index == len(flow.path) - 1):
                    # reach the dst, check jitter constraint.
                    # force gating enable if jitter constraint is not satisfied.
                    accumulated_jitter = latest_time - earliest_time
                    if accumulated_jitter > flow.jitter:
                        raise SchedulingError(ErrorType.JitterExceed,
                                              "Invalid due to jitter constraint.")

                gating_time = latest_time if gating else None
                end_time = latest_time + link.transmission_time(flow.payload)

                if end_time > flow.period:
                    raise SchedulingError(ErrorType.PeriodExceed,
                                          "Fail to find a valid solution.")

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

            if not scheduled:
                raise SchedulingError(ErrorType.PeriodExceed, "Fail to find a valid solution.")

            if gating:
                try:
                    link.add_gating(flow.period)
                except RuntimeError:
                    raise SchedulingError(ErrorType.GatingExceed,
                                          "Invalid due to gating constraint.")

            self.reward += 0.1

        except SchedulingError as e:
            logging.debug(e)
            if e.error_type == ErrorType.AlreadyScheduled:
                done = False
            elif e.error_type == ErrorType.JitterExceed:
                self.reward -= 100
                done = True
            elif e.error_type == ErrorType.GatingExceed:
                self.reward -= 100
                done = True
            elif e.error_type == ErrorType.PeriodExceed:
                self.reward -= 100
                done = True
            else:
                assert False, "Unknown error type."
            return self._generate_state(), self.reward, done, False, {}

        done = False
        # successfully scheduling
        if len(flow.path) == hop_index + 1:
            # reach the dst
            self.flows_scheduled[flow_index] = 1

            if all(self.flows_scheduled):
                # all flows are scheduled.
                done = True
                logging.info("Good job! Finish scheduling!")
                self.reward += 100

        self.render()
        return self._generate_state(), self.reward, done, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        logging.debug(self.flows_scheduled)
        logging.debug(self.last_action)

        flow_index, gating = self.last_action

        logging.debug(self.flows_operations[self.flows[flow_index]][-1])
        return

    def close(self):
        return
