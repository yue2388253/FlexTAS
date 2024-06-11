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
import random
from typing import SupportsFloat, Any, Optional

from definitions import ROOT_DIR, OUT_DIR, LOG_DIR
from src.lib.graph import neighbors_within_distance
from src.lib.operation import Operation, check_operation_isolation
from src.network.net import Flow, Link, Net, PERIOD_SET, generate_cev, generate_flows, Network


MAX_NEIGHBORS = 20
MAX_REMAIN_HOPS = 10


class ErrorType(Enum):
    JitterExceed = auto()
    PeriodExceed = auto()
    GatingExceed = auto()


class SchedulingError(Exception):
    def __init__(self, error_type: ErrorType, msg):
        super().__init__(f"SchedulingError: {msg}")
        self.error_type: ErrorType = error_type
        self.msg: str = msg


class _StateEncoder:
    def __init__(self, env: 'NetEnv'):
        self.env = env
        self.max_neighbors = MAX_NEIGHBORS
        self.max_remain_hops = MAX_NEIGHBORS

        flows = self.env.flows
        self.periods_list = PERIOD_SET
        self.periods_list.sort()
        self.periods_one_hot_dict = pd.get_dummies(self.periods_list)
        self.periods_dict = {period: 0 for period in self.periods_list}

        link_dict = self.env.link_dict

        # [link, [period, num_flows]]
        self.link_flow_period_dict: dict = defaultdict(Counter)
        self.link_num_flows = defaultdict(int)
        for flow in flows:
            path = flow.path
            for link_id in path:
                link = link_dict[link_id]
                self.link_num_flows[link] += 1
                self.link_flow_period_dict[link][flow.period] += 1

        # pre-compute the neighbors for all links, to avoid heavy and duplicate computation during training
        self.neighbors_dict = {
            link: [link] + neighbors_within_distance(self.env.line_graph, link, 2)
            for link in link_dict
        }

        state = self.state()

        self.observation_space = spaces.Dict({
            "flow_feature": spaces.Box(low=0, high=1, shape=state['flow_feature'].shape, dtype=np.float32),
            "link_feature": spaces.Box(low=0, high=1, shape=state['link_feature'].shape, dtype=np.float32),
            "adjacency_matrix": spaces.Box(low=-1, high=self.max_neighbors,
                                           shape=state['adjacency_matrix'].shape,
                                           dtype=np.int64),
            "features_matrix": spaces.Box(low=0, high=1, shape=state['features_matrix'].shape, dtype=np.float32),
            "remain_hops": spaces.Box(low=0, high=1, shape=state['remain_hops'].shape, dtype=np.float32)
        })

    def _link_feature(self, link_id):
        link = self.env.link_dict[link_id]

        link_utilization = 0
        if len(self.env.links_operations[link]) != 0:
            for flow, operation in self.env.links_operations[link]:
                link_utilization += (operation.end_time - operation.start_time) / flow.period
        assert link_utilization <= 1

        if self.link_num_flows[link] == 0:
            # no flow in this link
            num_flows_schedueld = 1
        else:
            num_flows_schedueld = len(self.env.links_operations[link])
            # normalized
            num_flows_schedueld /= self.link_num_flows[link]
        assert 0 <= num_flows_schedueld <= 1

        link_gcl_feature = np.array([
            math.sqrt(link_utilization),    # do sqrt operation since this value is always very small
            link.gcl_cycle / Net.GCL_CYCLE_MAX,
            link.gcl_length / Net.GCL_LENGTH_MAX,
            num_flows_schedueld
        ], dtype=np.float32)

        link_flow_periods_feature = np.array([
            self.link_flow_period_dict[link][period] / len(self.env.flows)
            for period in self.periods_list
        ])

        feature = np.concatenate((
            link_flow_periods_feature, link_gcl_feature
        ), dtype=np.float32)

        return feature

    def _flow_feature(self):
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
                flow.payload / Net.MTU,
                flow.jitter / flow.period,
                min(1, accum_jitter / flow.jitter) if flow.jitter != 0 else int(accum_jitter > 0),
                (hop_index + 1) / len(flow.path)
            ]
        ], dtype=np.float32)
        return flow_feature

    def _neighbors_features(self, current_link):
        neighbors = self.neighbors_dict[current_link]

        if len(neighbors) > self.max_neighbors:
            neighbors = neighbors[:self.max_neighbors]  # Truncate to max_neighbors
        elif len(neighbors) < self.max_neighbors:
            neighbors += [-1] * (self.max_neighbors - len(neighbors))  # Pad with -1 or another invalid index

        # Feature matrix and adjacency matrix handling
        feature_matrix = []
        edges = []
        max_edges = self.max_neighbors * (self.max_neighbors - 1)
        for idx, link_id in enumerate(neighbors):
            if link_id != -1:
                feature = self._link_feature(link_id)
            else:
                # Padding node: it must not be the first one.
                feature = np.zeros_like(feature_matrix[-1])

            feature_matrix.append(feature)

            for jdx, dst_link in enumerate(neighbors):
                if dst_link == -1 or not self.env.line_graph.has_edge(link_id, dst_link):
                    continue
                edges.append([idx, jdx])

        if len(edges) < max_edges:
            # Pad edge_index to ensure consistent shape
            padded_edges = edges + [[-1, -1]] * (max_edges - len(edges))  # Pad with non-existent edge
        else:
            padded_edges = edges[:max_edges]  # Ensure it does not exceed max_edges

        edge_index = np.array(padded_edges, dtype=np.int64).T
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        return edge_index, feature_matrix

    def _remain_nodes_features(self, flow, hop_index):
        path = flow.path
        features = []
        for i, link in enumerate(path):
            if i < hop_index:
                continue
            features.append(self._link_feature(link))

        # features must not be empty
        assert len(features) > 0

        # padding
        while len(features) < self.max_remain_hops:
            features.append(np.zeros_like(features[-1]))

        # truncate
        if len(features) > self.max_remain_hops:
            features = features[:self.max_remain_hops]

        # flatten the features
        features = np.array(features, dtype=np.float32).ravel()

        return features

    def state(self):
        flow = self.env.flows[self.env.flow_index]
        hop_index = len(self.env.flows_operations[flow])
        current_link = flow.path[hop_index]

        flow_feature = self._flow_feature()
        link_feature = self._link_feature(current_link)
        edge_index, feature_matrix = self._neighbors_features(current_link)
        remain_hops_feature = self._remain_nodes_features(flow, hop_index)

        return {
            "flow_feature": flow_feature,
            "link_feature": link_feature,
            "adjacency_matrix": edge_index,
            "features_matrix": feature_matrix,
            "remain_hops": remain_hops_feature
        }


class NetEnv(gym.Env):
    alpha: float = 1
    beta: float = 10
    gamma: float = 0.1

    def __init__(self, network:Network=None):
        super().__init__()

        if network is None:
            graph = generate_cev()
            network = Network(graph, generate_flows(graph, 10))
        
        self.graph = network.graph
        self.flows = network.flows
        self.line_graph, self.link_dict = network.line_graph, network.links_dict

        assert self.graph is not None and self.flows is not None, "fail to init env, invalid graph or flows"

        self.num_flows: int = len(self.flows)

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

        # shuffle the flows, thus
        #  a) to avoid local minimum,
        #  b) for generalization
        #  c) for robustness
        #  d) for learning efficiency,
        #  etc.
        random.shuffle(self.flows)
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

        for flow_rhs, operation_rhs in self.links_operations[link][:-1]:
            offset = check_operation_isolation(
                (operation, flow.period),
                (operation_rhs, flow_rhs.period)
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

        try:
            hop_index = len(self.flows_operations[flow])

            link = self.link_dict[flow.path[hop_index]]

            trans_time = link.transmission_time(flow.payload)

            # compute enqueue_time min and max
            if hop_index == 0:
                earliest_enqueue_time = 0
                latest_enqueue_time = 0
            else:
                last_link, last_operation = self.flows_operations[flow][hop_index - 1]

                is_gating_last_link = self.last_action
                if is_gating_last_link:
                    earliest_dequeue_time = last_operation.gating_time
                    latest_dequeue_time = last_operation.gating_time
                else:
                    earliest_dequeue_time = last_operation.earliest_time
                    latest_dequeue_time = last_operation.latest_time

                earliest_enqueue_time = (earliest_dequeue_time
                                         + trans_time
                                         + Net.DELAY_PROP
                                         - Net.SYNC_PRECISION
                                         + Net.DELAY_PROC_MIN)

                latest_enqueue_time = (latest_dequeue_time
                                       + trans_time
                                       + Net.DELAY_PROP
                                       + Net.SYNC_PRECISION
                                       + Net.DELAY_PROC_MAX)

            # construct operation
            if gating:
                wait_time = 0   # no-wait
            else:
                wait_time = link.interference_time()    # might wait
            latest_dequeue_time = latest_enqueue_time + wait_time

            if hop_index == len(flow.path) - 1:
                # reach the dst, check jitter constraint.
                if not gating:
                    # don't need to check if gating, since gating reset the jitter.
                    accumulated_jitter = latest_enqueue_time - earliest_enqueue_time
                    if accumulated_jitter > flow.jitter:
                        raise SchedulingError(ErrorType.JitterExceed,
                                              "jitter constraint unsatisfied.")

            end_time = latest_dequeue_time + trans_time

            if end_time > flow.period:
                raise SchedulingError(ErrorType.PeriodExceed,
                                      "injection time is too late")

            operation = Operation(
                earliest_enqueue_time,
                None,
                latest_dequeue_time,
                end_time
            )
            if gating:
                operation.gating_time = latest_dequeue_time    # always enable gating right after the latest enqueue time.

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
            self.reward = (1
                           - self.alpha * gcl_added / link.gcl_capacity
                           - self.beta * wait_time / flow.e2e_delay)

        except SchedulingError as e:
            self.logger.info(f"end of episode, reason: [{e.error_type}: {e.msg}]\tScheduled flows: {sum(self.flows_scheduled)}")
            if e.error_type == ErrorType.JitterExceed:
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
        # successfully scheduling a flow
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

                return (self.observation_space.sample(), self.reward, True, False,
                        {'success': True, 'ScheduleRes': self.links_operations.copy()})

        self.last_action = gating

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
    Once the agent has learnt how to schedule current flows, generate a new flow set for training.
    """
    def __init__(self, graph, flow_generator, num_flows,
                 initial_ratio=0.2, step_ratio=0.05, changing_freq=10):

        self.flow_generator = flow_generator

        self.num_flows_target = num_flows

        # the number of flows newly added each time changing the env.
        self.num_flows_step = math.ceil(num_flows * step_ratio)

        # start with half of the target num_flows and incrementally add flows if agent has learnt to schedule.
        num_flows_initial = math.ceil(num_flows * initial_ratio)
        flows = flow_generator(num_flows_initial)

        super().__init__(Network(graph, flows))

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
                flows = self.flow_generator(num_flows)
                super().__init__(Network(self.graph, flows))
                self.logger.info(f"Great! The agent has already learnt how to solve the problem. "
                                 f"Change the flows to train the agent. num_flows: {num_flows}")

                self.num_passed = 0

        return res
