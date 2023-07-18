from collections import defaultdict, Counter
from enum import Enum, auto
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
import logging
import numpy as np
import networkx as nx
import os
import pandas as pd
from typing import SupportsFloat, Any, Optional

from definitions import ROOT_DIR, OUT_DIR
from src.network.net import Duration, Flow, Link, transform_line_graph, Net, PERIOD_SET
from src.network.from_json import generate_net_flows_from_json
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

        self.graph = nx.convert_node_labels_to_integers(self.env.line_graph, label_attribute='link_id')

        flows = self.env.flows
        self.periods_list = PERIOD_SET
        self.periods_list.sort()
        self.periods_one_hot_dict = pd.get_dummies(self.periods_list)
        self.periods_dict = {period: 0 for period in self.periods_list}

        link_dict = self.env.link_dict
        self.links_one_hot_dict = pd.get_dummies(link_dict.keys())

        # [link, [period, num_flows]]
        self.link_flow_period_dict: dict = defaultdict(Counter)
        for flow in flows:
            path = flow.path
            for link_id in path:
                link = link_dict[link_id]
                self.link_flow_period_dict[link][flow.period] += 1

        self.edge_lists = np.array(self.graph.edges, dtype=np.int64).T

        state = self.state()
        self.observation_space = spaces.Dict({
            "flow_feature": spaces.Box(low=0, high=1, shape=state['flow_feature'].shape, dtype=np.float32),
            "link_feature": spaces.Box(low=0, high=1, shape=state['link_feature'].shape, dtype=np.float32),
            "adjacency_matrix": spaces.Box(low=0, high=len(self.graph.nodes)-1, shape=(2, len(self.graph.edges)), dtype=np.int64),
            "features_matrix": spaces.Box(low=0, high=1, shape=state['features_matrix'].shape, dtype=np.float32)
        })

    def state(self):
        flow = self.env.flows[self.env.flow_index]

        flow_feature = np.concatenate([
            self.periods_one_hot_dict[flow.period],
            [
                flow.period / Net.GCL_CYCLE_MAX,
                flow.payload / Net.PAYLOAD_MAX,
                flow.jitter / flow.period
            ]
        ], dtype=np.float32)

        link = flow.path[len(self.env.flows_operations[flow])]
        link_feature = np.array(self.links_one_hot_dict[link], dtype=np.float32)

        # the shape would be (num_operations+2, num_links) after encoding
        feature_matrix = []

        for node in self.graph.nodes:
            link_id = self.graph.nodes[node]['link_id']
            link = self.env.link_dict[link_id]

            link_gcl_feature = np.concatenate([
                self.links_one_hot_dict[link_id],
                [
                    link.gcl_cycle / Net.GCL_CYCLE_MAX,
                    link.gcl_length / Net.GCL_LENGTH_MAX
                ]
            ])

            link_flow_periods_feature = np.array([
                self.link_flow_period_dict[link][period] / len(self.env.flows)
                for period in self.periods_list
            ])

            feature = np.concatenate((
                link_flow_periods_feature, link_gcl_feature
            ))

            feature_matrix.append(feature)

        feature_matrix = np.array(feature_matrix, dtype=np.float32)

        return {
            "flow_feature": flow_feature,
            "link_feature": link_feature,
            "adjacency_matrix": self.edge_lists,
            "features_matrix": feature_matrix
        }


class NetEnv(gym.Env):
    alpha: float = 1
    beta: float = 10

    def __init__(self, graph: nx.DiGraph = None, flows: list[Flow] = None):
        super().__init__()

        if graph is None and flows is None:
            self.graph, self.flows = generate_net_flows_from_json(os.path.join(ROOT_DIR, 'data/input/smt_output.json'))
        elif graph is not None and flows is not None:
            self.graph: nx.Graph = graph
            self.flows: list[Flow] = flows

        assert self.graph is not None and self.flows is not None, "fail to init env, invalid graph or flows"

        self.num_flows: int = len(self.flows)
        self.line_graph: nx.Graph
        self.link_dict: dict[str, Link]
        self.line_graph, self.link_dict = transform_line_graph(self.graph)

        self.flows_operations: dict[Flow, list[tuple[Link, Operation]]] = defaultdict(list)
        self.links_operations: dict[Link, list[tuple[Flow, Operation]]] = defaultdict(list)

        self.flows_scheduled: list[int] = [0 for _ in range(self.num_flows)]

        self.flow_index: int = 0

        self.last_action = None

        self.reward: float = 0

        self.state_encoder: _StateEncoder = _StateEncoder(self)

        self.observation_space: spaces.Dict = self.state_encoder.observation_space

        # action space: enable gating or not for current operation
        self.action_space = spaces.Discrete(2)

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

        self.flow_index: int = 0

        self.reward = 0

        return self._generate_state(), {}

    def _generate_state(self) -> ObsType:
        return self.state_encoder.state()

    def action_masks(self) -> list[int]:
        # todo: disable gating if it will make the GCL capacity exceeded.
        return [True, True]

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
        """

        :param action:
        :return:
        tuple: A tuple containing the following elements:
            - observation (object): The new state of the environment after the action.
            - reward (float): The reward for the action.
            - done (bool): A flag indicating whether the game has ended. True means the game has ended.
            - truncated (bool): always False.
            - info (dict): A dictionary with extra diagnostic information.
                'success' key indicates whether the game has been successfully completed.
                True means success, False means failure.
                'msg' key contains information for debug
        """
        gating = (action == 1)

        flow = self.flows[self.flow_index]

        self.last_action = gating

        try:
            hop_index = len(self.flows_operations[flow])

            link = self.link_dict[flow.path[hop_index]]

            wait_time = 0
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

                if not gating:
                    wait_time = Net.DELAY_INTERFERENCE
                    latest_time += wait_time

                if (not gating) and (hop_index == len(flow.path) - 1):
                    # reach the dst, check jitter constraint.
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

            while True:
                offset = self.check_valid_flow(flow)
                if offset is None:
                    # find a valid solution that satisfies timing constraint
                    break

                assert isinstance(offset, int)

                for link, operation in self.flows_operations[flow]:
                    operation.add(offset)
                    if operation.end_time > flow.period:
                        # cannot be scheduled
                        raise SchedulingError(ErrorType.PeriodExceed, "Fail to find a valid solution.")

            gcl_added = 0
            if gating:
                # check gating constraint
                try:
                    old_gcl = link.gcl_length
                    link.add_gating(flow.period)
                    new_gcl = link.gcl_length
                    gcl_added = new_gcl - old_gcl
                except RuntimeError:
                    raise SchedulingError(ErrorType.GatingExceed,
                                          "Invalid due to gating constraint.")

            # todo: re-design the reward function, refer to the paper for more info.
            # self.reward += 0.1
            self.reward += 1 - self.alpha * gcl_added / link.max_gcl_length - self.beta * wait_time / flow.e2e_delay

        except SchedulingError as e:
            logging.info(f"{e}\nScheduled flows num: {sum(self.flows_scheduled)},\t"
                         f"Operations num: {sum([len(value) for value in self.flows_operations.values()])}")
            if e.error_type == ErrorType.AlreadyScheduled:
                done = False
            elif e.error_type == ErrorType.JitterExceed:
                # self.reward -= 100
                done = True
            elif e.error_type == ErrorType.GatingExceed:
                # self.reward -= 100
                done = True
            elif e.error_type == ErrorType.PeriodExceed:
                # self.reward -= 100
                done = True
            else:
                assert False, "Unknown error type."
            return self.observation_space.sample(), self.reward, done, False, {'success': False, 'msg': e.__str__()}

        done = False
        # successfully scheduling
        if len(flow.path) == hop_index + 1:
            # reach the dst
            self.flows_scheduled[self.flow_index] = 1
            self.flow_index += 1

            if all(self.flows_scheduled):
                # all flows are scheduled.
                filename = os.path.join(OUT_DIR, f'schedule_rl_{id(self)}.log')
                self.save_results(filename)
                logging.info(f"Good job! Finish scheduling! Scheduling result is saved at {filename}.")
                # self.reward += 100
                return self.observation_space.sample(), self.reward, True, False, {'success': True}

        self.render()
        return self._generate_state(), self.reward, done, False, {'success': done}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        gating = self.last_action
        logging.debug(f"Action: {gating}, Reward: {self.reward}")
        return

    def save_results(self, filename: str | os.PathLike):
        """
        Save the scheduling result to the dir, should only be called after all flows are scheduling.
        :return:
        """
        assert all(self.flows_scheduled), "Flows are not yet scheduled, cannot save results."

        res = []
        for flow, link_operations in self.flows_operations.items():
            res.append(
                str(len(res)) + '. ' + str(flow) + '\n'
                + '\n'.join([f"\t{link.link_id}, {operation}" for link, operation in link_operations])
                + '\n'
            )

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.writelines(res)

    def close(self):
        return
