from collections import defaultdict
import logging
import math
import networkx as nx
import os.path
from typing import Optional
import z3

from definitions import OUT_DIR
from src.lib.timing_decorator import timing_decorator
from src.network.net import Flow, Link, Net, Network
from src.app.scheduler import BaseScheduler


class Oliver2018Scheduler(BaseScheduler):
    def __init__(self, network: Network, **kwargs):
        super().__init__(network, **kwargs)

        self.hyper_period = math.lcm(*[flow.period for flow in self.flows])
        assert self.hyper_period <= Net.GCL_CYCLE_MAX

        self.links_variable = {link: self.LinkVariables(link) for link in self.links_dict.values()}

        self.links_streams = defaultdict(list)
        for flow in self.flows:
            for link_id in flow.path:
                self.links_streams[self.links_dict[link_id]].append(flow)

        # key: flow -> link
        self.streams_instances = defaultdict(lambda: defaultdict(list))
        for flow in self.flows:
            for hop_id, link_id in enumerate(flow.path):
                link = self.links_dict[link_id]
                for instance_id in range(math.ceil(self.hyper_period / flow.period)):
                    self.streams_instances[flow][link].append(
                        self.StreamInstance(
                            stream_id=flow.flow_id,
                            link_id=link_id,
                            instance_id=instance_id,
                            hop_id=hop_id
                        )
                    )

        self.constraints_set = []
        self.model: Optional[z3.ModelRef] = None

        assert isinstance(self.timeout_s, int)
        # the `timeout` param of z3 is in millisecond unit.
        z3.set_param("timeout", self.timeout_s * 1000)

    class LinkVariables:
        def __init__(self, link: Link):
            # 下列三个数组是需要SMT求解的未知变量
            # 变量命名规则：
            # Phi/Tau/Kappa^(link_id)： P/T/K区分三个array
            #           i表示link_id
            # 开窗口的时刻的array，即论文中的Phi
            self.phi_array = z3.Array(f'P^({link.link_id})', z3.IntSort(), z3.IntSort())
            # 关窗口的时刻的array，即论文中的Tau
            self.tau_array = z3.Array(f'T^({link.link_id})', z3.IntSort(), z3.IntSort())
            # 初始的tau
            self.tau_0_array = z3.Array(f'T_0^({link.link_id})', z3.IntSort(), z3.IntSort())
            # 存放中间结果的tau
            self.tau_1_array = z3.Array(f'T_1^({link.link_id})', z3.IntSort(), z3.IntSort())
            # 窗口到队列的映射，即论文中的Kappa
            self.kappa_array = z3.Array(f'K^({link.link_id})', z3.IntSort(), z3.IntSort())

    class StreamInstance:
        # Stream_Instance包含以下几个成员变量
        # stream_id, link_id, instance_id, Omega(window_index)
        # 其中，Omega是未知数，需要z3求解
        def __init__(self, stream_id, link_id,
                     instance_id, hop_id):
            self.stream_id = stream_id
            self.link_id = link_id
            self.instance_id = instance_id
            # 该流实例对应这条流的哪一跳路由
            self.hop_id = hop_id
            # 即该报文实例对应在Phi和Tau两个array内的索引
            # 对应论文中的Omega
            # Omega也是一个未知数，在初始化的阶段再分配变量名
            # name = 'Omega_' + str(self.stream_id) + ',' + str(self.link_id) \
            #        + '_' + str(self.instance_id)
            self.omega = z3.Int(f'O_{stream_id},{instance_id}^({link_id})')

    def _construct_constraints(self):
        logging.info("Constructing constraints...")
        # 1. well-defined windows constraints
        for link_id, link in self.links_dict.items():
            if link in self.links_streams:
                phi = self.links_variable[link].phi_array[0]
                tau = self.links_variable[link].tau_array[link.gcl_capacity - 1]
                constraint_1 = z3.And(phi >= 0, tau < self.hyper_period)
                # print(constraint_1)
                self.constraints_set.append(constraint_1)

        for link_id, link in self.links_dict.items():
            if link in self.links_streams:
                for k in range(link.gcl_capacity):
                    kappa = self.links_variable[link].kappa_array[k]
                    constraint_1 = z3.And(kappa >= 0, kappa < Net.ST_QUEUES)
                    self.constraints_set.append(constraint_1)

        # 2. stream instance constraints
        for flow, flow_instances in self.streams_instances.items():
            for link, flow_link_instances in flow_instances.items():
                j = 0
                for instance_obj in flow_link_instances:
                    link_id = instance_obj.link_id
                    period = flow.period
                    phi = self.links_variable[link].phi_array
                    tau = self.links_variable[link].tau_array
                    omega = instance_obj.omega
                    constraint_2 = z3.And(phi[omega] >= j * period,
                                          tau[omega] < (j + 1) * period)
                    self.constraints_set.append(constraint_2)
                    j += 1

        # 3. ordered windows constraint
        for link_id, link in self.links_dict.items():
            gcl_len = link.gcl_capacity
            for i in range(gcl_len - 1):
                phi = self.links_variable[link].phi_array[i+1]
                tau = self.links_variable[link].tau_array[i]
                constraint_3 = (tau <= phi)
                self.constraints_set.append(constraint_3)

        # 4. frame-to-window assignment constraint
        for flow, flow_instances in self.streams_instances.items():
            for link, flow_link_instances in flow_instances.items():
                for instance_obj in flow_link_instances:
                    gcl_len = link.gcl_capacity
                    omega = instance_obj.omega
                    constraint_4 = z3.And(omega >= 0, omega <= gcl_len - 1)
                    self.constraints_set.append(constraint_4)

        # 5. window size constraint
        for link_id, link in self.links_dict.items():
            gcl_len = link.gcl_capacity
            phi_array = self.links_variable[link].phi_array
            tau_0_array = self.links_variable[link].tau_0_array
            if link not in self.links_streams:
                continue
            for k in range(gcl_len):
                # Store(a, i, v) returns a new array identical to a,
                # but on position i it contains the value v
                # array操作无法添加到solver里面？
                # tau = Store(tau, k, phi[k])
                # T_0的值应当永远与phi保持一致
                # s.add(tau_0_array == Store(tau_0_array, k, phi_array[k]))
                constraint_5 = (tau_0_array == z3.Store(tau_0_array, k, phi_array[k]))
                self.constraints_set.append(constraint_5)

        # 将tau_1所有元素初始化成0
        for link_id, link in self.links_dict.items():
            gcl_len = self.links_dict[link_id].gcl_capacity
            tau_1_array = self.links_variable[link].tau_1_array
            if link not in self.links_streams:
                continue
            for k in range(gcl_len):
                tau_1_array = z3.Store(tau_1_array, k, 0)
                self.links_variable[link].tau_1_array = tau_1_array

        # 用tau_1存放中间结果
        for flow, flow_instances in self.streams_instances.items():
            for link, flow_link_instances in flow_instances.items():
                for stream_instance_obj in flow_link_instances:
                    tau_1_array = self.links_variable[link].tau_1_array
                    omega = stream_instance_obj.omega
                    trans_duration = link.transmission_time(flow.payload)
                    tau_1_array = z3.Store(tau_1_array, omega, tau_1_array[omega] + trans_duration)
                    self.links_variable[link].tau_1_array = tau_1_array

        # 给tau添加约束
        for link in self.links_dict.values():
            gcl_len = link.gcl_capacity
            tau_array = self.links_variable[link].tau_array
            tau_0_array = self.links_variable[link].tau_0_array
            tau_1_array = self.links_variable[link].tau_1_array
            if link not in self.links_streams:
                continue
            for k in range(gcl_len):
                constraint_5 = (tau_array == z3.Store(tau_array, k, tau_0_array[k] + tau_1_array[k]))
                self.constraints_set.append(constraint_5)

        # 6.    stream constraint
        #       同一个报文在相邻两跳之间的先后顺序
        sync_precision = Net.SYNC_PRECISION
        for flow, flow_instances in self.streams_instances.items():
            route_path_len = len(flow.path)
            for k in range(route_path_len - 1):
                link = self.links_dict[flow.path[k]]
                suc_link = self.links_dict[flow.path[k + 1]]
                instance_num_in_hyper_period = len(flow_instances[link])
                for j in range(instance_num_in_hyper_period):
                    pre_tau = self.links_variable[link].tau_array
                    suc_phi = self.links_variable[suc_link].phi_array
                    pre_omega = self.streams_instances[flow][link][j].omega
                    suc_omega = self.streams_instances[flow][suc_link][j].omega
                    constraint_6 = (pre_tau[pre_omega] + sync_precision <= suc_phi[suc_omega])
                    self.constraints_set.append(constraint_6)

        # 7. stream isolation constraint
        # 描述的是在一条链路上汇聚的任意两个帧实例之间的关系
        # loop1：   对于每一条链路，找到属于这一条链路的所有流量的集合，
        #           并记录这些流量在其哪一跳经过该链路，用于索引route[]，
        #           方便找到该流量的上一跳
        for link_id, link in self.links_dict.items():

            # 当前链路的tau数组
            ab_tau_array = self.links_variable[link].tau_array
            # 当前链路的phi数组
            ab_phi_array = self.links_variable[link].phi_array
            # 当前链路的kappa数组
            ab_kappa_array = self.links_variable[link].kappa_array

            stream_num = len(self.links_streams[link])
            for i in range(stream_num):
                for j in range(i + 1, stream_num):
                    i_stream = self.links_streams[link][i]
                    j_stream = self.links_streams[link][j]

                    i_ab_hop_id = i_stream.path.index(link_id)
                    j_ab_hop_id = j_stream.path.index(link_id)

                    i_stream_instance_obj_set_at_ab_hop = self.streams_instances[i_stream][link]
                    j_stream_instance_obj_set_at_ab_hop = self.streams_instances[j_stream][link]

                    if i_ab_hop_id == 0 and j_ab_hop_id == 0:
                        for k_ab_instance_obj in i_stream_instance_obj_set_at_ab_hop:
                            for l_ab_instance_obj in j_stream_instance_obj_set_at_ab_hop:
                                i_k_omega = k_ab_instance_obj.omega
                                j_l_omega = l_ab_instance_obj.omega

                                constraint_7 = z3.Or(ab_tau_array[i_k_omega] + sync_precision <=
                                                     ab_tau_array[j_l_omega],
                                                     ab_tau_array[j_l_omega] + sync_precision <=
                                                     ab_phi_array[i_k_omega],
                                                     ab_kappa_array[i_k_omega] != ab_kappa_array[j_l_omega],
                                                     i_k_omega == j_l_omega)
                                self.constraints_set.append(constraint_7)

                    else:
                        # 如果当前链路不是起源于talker的链路
                        # 这意味着这些流有各自的上一跳节点，因此需要添加四个式子
                        i_xa_link = self.links_dict[i_stream.path[i_ab_hop_id - 1]]
                        j_ya_link = self.links_dict[j_stream.path[j_ab_hop_id - 1]]

                        i_xa_hop_phi_array = self.links_variable[i_xa_link].phi_array
                        j_ya_hop_phi_array = self.links_variable[j_ya_link].phi_array

                        i_stream_instance_obj_set_at_xa_hop = self.streams_instances[i_stream][i_xa_link]
                        j_stream_instance_obj_set_at_ya_hop = self.streams_instances[j_stream][j_ya_link]

                        for (k_ab_instance_obj, k_xa_instance_obj) in \
                                zip(i_stream_instance_obj_set_at_ab_hop, i_stream_instance_obj_set_at_xa_hop):
                            for (l_ab_instance_obj, l_ya_instance_obj) in \
                                    zip(j_stream_instance_obj_set_at_ab_hop, j_stream_instance_obj_set_at_ya_hop):
                                i_k_ab_omega = k_ab_instance_obj.omega
                                j_l_ab_omega = l_ab_instance_obj.omega
                                i_k_xa_omega = k_xa_instance_obj.omega
                                j_l_ya_omega = l_ya_instance_obj.omega

                                constraint_7 = z3.Or(ab_tau_array[i_k_ab_omega] + sync_precision <=
                                                     j_ya_hop_phi_array[j_l_ya_omega],
                                                     ab_tau_array[j_l_ab_omega] + sync_precision <=
                                                     i_xa_hop_phi_array[i_k_xa_omega],
                                                     ab_kappa_array[i_k_ab_omega] != ab_kappa_array[j_l_ab_omega],
                                                     i_k_ab_omega == j_l_ab_omega)

                                self.constraints_set.append(constraint_7)

        # 8. stream end-to-end latency constraint
        for flow, flow_instances in self.streams_instances.items():

            first_link = self.links_dict[flow.path[0]]
            last_link = self.links_dict[flow.path[-1]]
            first_phi_array = self.links_variable[first_link].phi_array
            last_tau_array = self.links_variable[last_link].tau_array
            latency_requirement = flow.e2e_delay

            instance_num_in_hyper_period = len(flow_instances[first_link])
            for i in range(instance_num_in_hyper_period):
                first_instance_obj = flow_instances[first_link][i]
                last_instance_obj = flow_instances[last_link][i]
                first_omega = first_instance_obj.omega
                last_omega = last_instance_obj.omega
                constraint_8 = (last_tau_array[last_omega] - first_phi_array[first_omega] <=
                                latency_requirement - sync_precision)
                self.constraints_set.append(constraint_8)

        # 9. stream jitter constraints
        # sender jitter
        for flow, flow_instances in self.streams_instances.items():
            link = self.links_dict[flow.path[0]]
            phi_array = self.links_variable[link].phi_array
            tau_array = self.links_variable[link].tau_array

            instance_obj_set_at_first_link = flow_instances[link]

            instance_num_in_hyper_period = len(flow_instances[link])
            period = flow.period
            trans_duration = link.transmission_time(flow.payload)
            jitter_requirement = flow.jitter

            for j in range(instance_num_in_hyper_period):
                for k in range(instance_num_in_hyper_period):
                    j_omega = instance_obj_set_at_first_link[j].omega
                    k_omega = instance_obj_set_at_first_link[k].omega

                    constraint_9 = ((tau_array[j_omega] - j * period) - (phi_array[k_omega] - k * period) -
                                    trans_duration <= jitter_requirement)
                    self.constraints_set.append(constraint_9)

        # receiver jitter
        for flow, flow_instances in self.streams_instances.items():
            last_link = self.links_dict[flow.path[-1]]
            phi_array = self.links_variable[last_link].phi_array
            tau_array = self.links_variable[last_link].tau_array

            instance_obj_set_at_last_link = flow_instances[last_link]
            instance_num_in_hyper_period = len(flow_instances[last_link])

            period = flow.period
            trans_duration = last_link.transmission_time(flow.payload)
            jitter_requirement = flow.jitter
            for j in range(instance_num_in_hyper_period):
                for k in range(instance_num_in_hyper_period):
                    j_omega = instance_obj_set_at_last_link[j].omega
                    k_omega = instance_obj_set_at_last_link[k].omega

                    constraint_9 = ((tau_array[j_omega] - j * period) - (phi_array[k_omega] - k * period) -
                                    trans_duration <= jitter_requirement)
                    self.constraints_set.append(constraint_9)

        logging.info(f"Total constraints: {len(self.constraints_set)} ")

    def _solve_constraints(self) -> bool:
        solver = z3.Solver()
        for constraint in self.constraints_set:
            solver.add(constraint)

        logging.info("Checking constraints...")
        is_sat = solver.check()

        if is_sat == z3.sat:
            self.model = solver.model()
            print("sat")
            logging.info(f"Successfully scheduled.")
            return True
        elif is_sat == z3.unsat:
            print("unsat")
            logging.error("z3 fail to find a valid solution.")
        elif is_sat == z3.unknown:
            print("unknown")
            logging.error(f"z3 unknown: {solver.reason_unknown()}")
        else:
            raise NotImplementedError
        return False

    def save_results(self, filename):
        pass

    def schedule(self) -> bool:
        self._construct_constraints()
        is_scheduled = self._solve_constraints()
        if is_scheduled:
            filename = os.path.join(OUT_DIR, f'Oliver2018_{id(self)}.log')
            self.save_results(filename)
            logging.info(f"The scheduling result is save at {filename}")
        return is_scheduled
