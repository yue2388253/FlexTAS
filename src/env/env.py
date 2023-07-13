from collections import defaultdict
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
from src.network.net import Duration, Flow, Link, transform_line_graph, Net
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

        flows = self.env.flows
        edges = []
        nodes = ["S", "T"]
        for flow_index, flow in enumerate(flows):
            path = flow.path
            nodes.append((flow_index, path[0]))
            edges.append(("S", (flow_index, path[0])))
            for i in range(len(path) - 1):
                nodes.append((flow_index, path[i+1]))
                edges.append(((flow_index, path[i]), (flow_index, path[i+1])))
            edges.append(((flow_index, path[len(path)-1]), "T"))

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        self.graph = graph

        # the shape would be (num_operations+2, num_operations+2)
        self.adjacency_matrix = np.array(nx.to_scipy_sparse_array(self.graph).todense(), dtype=np.float32)

        self.num_operations = sum([len(flow.path) for flow in self.env.flows])
        state = self.state()
        self.observation_space = spaces.Dict({
            "adjacency_matrix": spaces.Box(low=0, high=1, shape=(self.num_operations+2, self.num_operations+2), dtype=np.float32),
            "features_matrix": spaces.Box(low=0, high=1, shape=state['features_matrix'].shape, dtype=np.float32)
        })

    def state(self):
        links_id = self.env.link_dict.keys()
        one_hot_dict = pd.get_dummies(links_id)
        flows = self.env.flows
        link_dict = self.env.link_dict

        # the shape would be (num_operations+2, num_links) after encoding
        feature_matrix = []

        num_links = len(self.env.link_dict)
        for node in self.graph.nodes:
            if node == "S" or node == "T":
                feature_matrix.append(np.zeros(num_links+2+7,))
                continue

            flow_index, link_id = node
            flow = flows[flow_index]
            link = link_dict[link_id]

            link_one_hot_feature = one_hot_dict[link_id].values

            link_gcl_feature = np.array([link.gcl_cycle / Net.GCL_CYCLE_MAX, link.gcl_length / Net.GCL_LENGTH_MAX])

            operation_feature = None
            operations = self.env.flows_operations[flow]
            for link, operation in operations:
                if link.link_id == link_id:
                    operation_feature = np.array([
                        1,  # indicate this operation has been scheduled
                        1 if operation.gating_time is not None else 0,  # indicate enable gating
                        operation.start_time / flow.period,
                        operation.earliest_time / flow.period,
                        operation.latest_time / flow.period,
                        operation.end_time / flow.period,
                        max((operation.latest_time - operation.earliest_time) / flow.jitter, 1)
                    ])
            if operation_feature is None:
                operation_feature = np.zeros(7, )

            feature = np.concatenate((
                link_one_hot_feature, link_gcl_feature, operation_feature
            ))

            feature_matrix.append(feature)

        feature_matrix = np.array(feature_matrix, dtype=np.float32)

        return {"adjacency_matrix": self.adjacency_matrix,
                "features_matrix": feature_matrix}


class NetEnv(gym.Env):
    def __init__(self, graph: nx.Graph = None, flows: list[Flow] = None):
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

        self.last_action = None

        self.reward: float = 0

        self.state_encoder: _StateEncoder = _StateEncoder(self)

        self.observation_space: spaces.Dict = self.state_encoder.observation_space

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

            if gating:
                # check gating constraint
                try:
                    link.add_gating(flow.period)
                except RuntimeError:
                    raise SchedulingError(ErrorType.GatingExceed,
                                          "Invalid due to gating constraint.")

            self.reward += 0.1

        except SchedulingError as e:
            logging.info(f"{e}\nScheduled flows num: {sum(self.flows_scheduled)},\t"
                         f"Operations num: {sum([len(value) for value in self.flows_operations.values()])}")
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
            return self._generate_state(), self.reward, done, False, {'success': False, 'msg': e.__str__()}

        done = False
        # successfully scheduling
        if len(flow.path) == hop_index + 1:
            # reach the dst
            self.flows_scheduled[flow_index] = 1

            if all(self.flows_scheduled):
                # all flows are scheduled.
                done = True
                filename = os.path.join(OUT_DIR, f'schedule_rl_{id(self)}.log')
                self.save_results(filename)
                logging.info(f"Good job! Finish scheduling! Scheduling result is saved at {filename}.")
                self.reward += 100

        self.render()
        return self._generate_state(), self.reward, done, False, {'success': done}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        flow_index, gating = self.last_action
        flow_id = self.flows[flow_index]
        logging.debug(f"Action: ({flow_id}, {gating}), {self.flows_operations[self.flows[flow_index]][-1]}")
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
