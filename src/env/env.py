from collections import defaultdict, Counter
from enum import Enum, auto
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
import logging
import math
import numpy as np
import networkx as nx
import os
import pandas as pd
from typing import SupportsFloat, Any, Optional

from definitions import ROOT_DIR, OUT_DIR, LOG_DIR
from src.network.net import Duration, Flow, Link, transform_line_graph, Net, PERIOD_SET, generate_cev, generate_flows
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
            "adjacency_matrix": spaces.Box(low=0, high=len(self.graph.nodes) - 1, shape=(2, len(self.graph.edges)),
                                           dtype=np.int64),
            "features_matrix": spaces.Box(low=0, high=1, shape=state['features_matrix'].shape, dtype=np.float32)
        })

    def state(self):
        flow = self.env.flows[self.env.flow_index]

        accum_jitter = 0
        if len(self.env.flows_operations[flow]) != 0:
            operation = self.env.flows_operations[flow][-1][1]
            accum_jitter = operation.latest_time - operation.start_time

        hop_index = len(self.env.flows_operations[flow])
        flow_feature = np.concatenate([
            self.periods_one_hot_dict[flow.period],
            [
                flow.period / Net.GCL_CYCLE_MAX,
                flow.payload / Net.PAYLOAD_MAX,
                flow.jitter / flow.period,
                min(1, accum_jitter / flow.jitter),
                (hop_index + 1) / len(flow.path)
            ]
        ], dtype=np.float32)

        current_link = flow.path[hop_index]

        feature_matrix = []

        link_feature = None

        for node in self.graph.nodes:
            link_id = self.graph.nodes[node]['link_id']
            link = self.env.link_dict[link_id]

            link_utilization = 0
            if len(self.env.links_operations[link]) != 0:
                for flow, operation in self.env.links_operations[link]:
                    link_utilization += (operation.end_time - operation.start_time) / flow.period
            assert link_utilization <= 1

            link_gcl_feature = np.concatenate([
                self.links_one_hot_dict[link_id],
                [
                    link_utilization,
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
            ), dtype=np.float32)

            if link_id == current_link:
                link_feature = feature

            feature_matrix.append(feature)

        feature_matrix = np.array(feature_matrix, dtype=np.float32)

        assert link_feature is not None, "Link feature is not represented."

        return {
            "flow_feature": flow_feature,
            "link_feature": link_feature,
            "adjacency_matrix": self.edge_lists,
            "features_matrix": feature_matrix
        }


class NetEnv(gym.Env):
    alpha: float = 1
    beta: float = 10
    gamma: float = 0.1

    def __init__(self, graph: nx.DiGraph = None, flows: list[Flow] = None):
        super().__init__()

        if graph is None and flows is None:
            self.graph = generate_cev()
            self.flows = generate_flows(self.graph, 10)
        elif graph is not None and flows is not None:
            self.graph: nx.DiGraph = graph
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

        logger = logging.getLogger(f"{__name__}.{os.getpid()}")
        logger.setLevel(logging.INFO)
        self.logger = logger

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
        flow = self.flows[self.flow_index]
        hop_index = len(self.flows_operations[flow])
        link = self.link_dict[flow.path[hop_index]]
        return [True, link.add_gating(flow.period, attempt=True)]

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
                    wait_time = link.interference_time()
                    latest_time += wait_time

                if (not gating) and (hop_index == len(flow.path) - 1):
                    # reach the dst, check jitter constraint.
                    accumulated_jitter = latest_time - earliest_time
                    if accumulated_jitter > flow.jitter:
                        raise SchedulingError(ErrorType.JitterExceed,
                                              "jitter constraint unsatisfied.")

                gating_time = latest_time if gating else None
                end_time = latest_time + link.transmission_time(flow.payload)

                if end_time > flow.period:
                    raise SchedulingError(ErrorType.PeriodExceed,
                                          "injection time is too late")

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
                        raise SchedulingError(ErrorType.PeriodExceed, "timing isolation constraint unsatisfied.")

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
                                          "gating constraint unsatisfied.")

            # self.reward += 0.1
            self.reward = 1 - self.alpha * gcl_added / link.max_gcl_length - self.beta * wait_time / flow.e2e_delay

        except SchedulingError as e:
            self.logger.info(f"{e}\tScheduled flows: {sum(self.flows_scheduled)}")
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

            if self.flow_index % math.ceil(self.num_flows * 0.1) == 0:
                # give an extra reward when the agent schedule another set of flows.
                self.reward += self.gamma * ((self.flow_index / self.num_flows) ** 2)

            if all(self.flows_scheduled):
                # all flows are scheduled.
                filename = os.path.join(OUT_DIR, f'schedule_rl_{id(self)}.log')
                self.save_results(filename)
                self.logger.info(f"Good job! Finish scheduling! Scheduling result is saved at {filename}.")

                return self.observation_space.sample(), self.reward, True, False, {'success': True}

        self.render()
        return self._generate_state(), self.reward, done, False, {'success': done}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        gating = self.last_action
        self.logger.debug(f"Action: {gating}, Reward: {self.reward}")
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


class TrainingNetEnv(NetEnv):
    """
    Use curriculum learning to help training.
    Begin with an easy environment that contains a small set of flows for agent to learn,
    increase the number of flows gradually to make the env harder if the agent can easily
    pass the current env.
    """
    def __init__(self, graph, flow_generator, num_flows,
                 initial_ratio=0.2, step_ratio=0.05, changing_freq=10):

        self.flow_generator = flow_generator

        self.num_flows_target = num_flows

        # the number of flows newly added each time changing the env.
        self.num_flows_step = math.ceil(num_flows * step_ratio)

        # start with half of the target num_flows and incrementally add flows if agent has learnt to schedule.
        num_flows_initial = math.ceil(num_flows * initial_ratio)
        flows = flow_generator(graph, num_flows_initial)

        super().__init__(graph, flows)

        self.num_passed = 0
        self.changing_freq = changing_freq

        log_file = os.path.join(LOG_DIR, f"training_env_{os.getpid()}.txt")
        fh = logging.FileHandler(filename=log_file)
        fh.setLevel(logging.DEBUG)
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # Add the formatter to the handler
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(f"Start training with {num_flows_initial} flows.")

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        res = super().step(action)

        done, info = res[2], res[-1]
        if done and info['success']:
            self.num_passed += 1
            self.logger.info(f"passed the job! ({self.num_passed})")

            if self.num_passed == self.changing_freq:
                num_flows = min(self.num_flows_target, self.num_flows + self.num_flows_step)
                flows = self.flow_generator(self.graph, num_flows)
                super().__init__(self.graph, flows)
                self.logger.info(f"Great! The agent has already learnt how to solve the problem. "
                                 f"Change the flows to train the agent. num_flows: {num_flows}")

                self.num_passed = 0

        return res
