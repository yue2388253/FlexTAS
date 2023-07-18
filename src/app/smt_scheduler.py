from collections import defaultdict
import logging
import math
import networkx as nx
import os.path
from typing import Optional
import z3

from definitions import OUT_DIR
from src.lib.timing_decorator import timing_decorator
from src.network.net import Flow, Link, Net
from src.app.scheduler import BaseScheduler


class SmtScheduler(BaseScheduler):
    def __init__(self, graph: nx.Graph, flows: list[Flow], timeout_s: int = None):
        super().__init__(graph, flows, timeout_s)

        self.num_queues = 1

        self.z3_variables_flow: dict[Flow, dict[str, z3.ArithRef]] = \
            defaultdict(dict)
        self.z3_variables_flow_link: dict[Flow, dict[Link, dict[str, z3.ArithRef]]] = \
            defaultdict(lambda: defaultdict(dict))
        self.z3_variables_links: dict[Link, dict[str, z3.ArithRef]] = \
            defaultdict(dict)

        self.constraints_set = []
        self.model: Optional[z3.ModelRef] = None

        if timeout_s is not None:
            assert isinstance(timeout_s, int)
            # the `timeout` param of z3 is in millisecond unit.
            z3.set_param("timeout", timeout_s * 1000)

    @timing_decorator(logging.info)
    def schedule(self) -> bool:
        self._init_z3_variables()
        self._construct_constraints()
        is_scheduled = self._solve_constraints()
        if is_scheduled:
            filename = os.path.join(OUT_DIR, f'smt_schedule_{id(self)}.log')
            self.save_results(filename)
            logging.info(f"The scheduling result is save at {filename}")
        return is_scheduled

    def _init_z3_variables(self):
        flow_variables = ['jitter', 'queue_index']
        flow_link_variables = ['t3_min', 't3_max', 't4_min', 't4_max', 't5_min', 't5_max',
                               'tb', 'rr']
        for flow in self.flows:
            flow_id = flow.flow_id
            self.z3_variables_flow[flow] = \
                {var: z3.Int(f"{flow_id}_{var}") for var in flow_variables}
            for link_id in flow.path:
                link = self.links_dict[link_id]
                self.z3_variables_flow_link[flow][link] = \
                    {var: z3.Int(f"{flow_id}_{link_id}_{var}") for var in flow_link_variables}
                self.z3_variables_flow_link[flow][link]['gc'] = \
                    z3.Bool(f"{flow_id}_{link_id}_gc")

        link_variables = ['gcl_cycle', 'gcl_length']
        for link_id, link in self.links_dict.items():
            self.z3_variables_links[link] = \
                {var: z3.Int(f"{link_id}_{var}") for var in link_variables}

    def _construct_constraints(self):
        for flow in self.flows:
            self.constraints_set.append(
                z3.And(
                    self.z3_variables_flow[flow]['queue_index'] >= 0,
                    self.z3_variables_flow[flow]['queue_index'] < self.num_queues
                )
            )

        for flow in self.flows:
            path = flow.path
            first_link_id = path[0]
            first_link = self.links_dict[first_link_id]
            self.constraints_set.append(
                self.z3_variables_flow_link[flow][first_link]['t4_min'] ==
                self.z3_variables_flow_link[flow][first_link]['t4_max']
            )
            self.constraints_set.append(
                self.z3_variables_flow_link[flow][first_link]['t4_min'] > 0
            )
            self.constraints_set.append(
                self.z3_variables_flow_link[flow][first_link]['t4_min'] < flow.period
            )
            self.constraints_set.append(
                self.z3_variables_flow_link[flow][first_link]['t3_min'] ==
                self.z3_variables_flow_link[flow][first_link]['t4_min']
            )
            self.constraints_set.append(
                self.z3_variables_flow_link[flow][first_link]['t4_min'] ==
                self.z3_variables_flow_link[flow][first_link]['t4_max']
            )

            last_link_id = path[-1]
            last_link = self.links_dict[last_link_id]

            # ddl constraint
            self.constraints_set.append(
                self.z3_variables_flow_link[flow][last_link]['t5_max'] + Net.DELAY_PROP -
                self.z3_variables_flow_link[flow][first_link]['t3_min'] <= flow.e2e_delay
            )

            # jitter constraint
            self.constraints_set.append(
                self.z3_variables_flow[flow]['jitter'] ==
                self.z3_variables_flow_link[flow][last_link]['t5_max'] -
                self.z3_variables_flow_link[flow][last_link]['t5_min']
            )
            self.constraints_set.append(
                self.z3_variables_flow[flow]['jitter'] <= flow.jitter
            )

            for hop in range(len(path) - 1):
                link_ax = self.links_dict[path[hop]]
                link_xb = self.links_dict[path[hop + 1]]
                # flow transmission constraint
                self.constraints_set.append(
                    self.z3_variables_flow_link[flow][link_ax]['t5_min'] + Net.DELAY_PROP + Net.DELAY_PROC_MIN ==
                    self.z3_variables_flow_link[flow][link_xb]['t3_min']
                )
                self.constraints_set.append(
                    self.z3_variables_flow_link[flow][link_ax]['t5_max'] + Net.DELAY_PROP + Net.DELAY_PROC_MAX ==
                    self.z3_variables_flow_link[flow][link_xb]['t3_max']
                )

            for link_id in path:
                link = self.links_dict[link_id]
                trans_delay = link.transmission_time(flow.payload)
                # Frame transmission constraint
                self.constraints_set.append(
                    self.z3_variables_flow_link[flow][link]['t5_min'] -
                    self.z3_variables_flow_link[flow][link]['t4_min'] == trans_delay
                )
                self.constraints_set.append(
                    self.z3_variables_flow_link[flow][link]['t5_max'] -
                    self.z3_variables_flow_link[flow][link]['t4_max'] == trans_delay
                )

                # forbid cross cycle
                self.constraints_set.append(
                    self.z3_variables_flow_link[flow][link]['t5_max'] < flow.period
                )
                self.constraints_set.append(
                    self.z3_variables_flow_link[flow][link]['t3_min'] > 0
                )

        # contention free constraint
        for i in range(len(self.flows) - 1):
            flow_i = self.flows[i]
            path_i = flow_i.path
            for j in range(i + 1, len(self.flows)):
                flow_j = self.flows[j]
                path_j = flow_j.path
                common_links = list(set(path_i) & set(path_j))
                hp = math.lcm(flow_i.period, flow_j.period)
                rr_i = hp // flow_i.period
                rr_j = hp // flow_j.period

                for link_id in common_links:
                    link = self.links_dict[link_id]
                    delay_ifg = link.safe_distance()

                    for alpha in range(rr_i + 5):
                        for beta in range(rr_j + 5):
                            self.constraints_set.append(
                                z3.If(
                                    self.z3_variables_flow[flow_i]['queue_index'] ==
                                    self.z3_variables_flow[flow_j]['queue_index'],
                                    z3.Or(
                                        self.z3_variables_flow_link[flow_i][link]['t3_min'] + alpha * flow_i.period >=
                                        self.z3_variables_flow_link[flow_j][link]['t5_max'] + beta * flow_j.period +
                                        delay_ifg,
                                        self.z3_variables_flow_link[flow_j][link]['t3_min'] + beta * flow_j.period >=
                                        self.z3_variables_flow_link[flow_i][link]['t5_max'] + alpha * flow_i.period +
                                        delay_ifg
                                    ),
                                    z3.Or(
                                        self.z3_variables_flow_link[flow_i][link]['tb'] + alpha * flow_i.period >=
                                        self.z3_variables_flow_link[flow_j][link]['t5_max'] + beta * flow_j.period +
                                        delay_ifg,
                                        self.z3_variables_flow_link[flow_j][link]['tb'] + beta * flow_j.period >=
                                        self.z3_variables_flow_link[flow_i][link]['t5_max'] + alpha * flow_i.period +
                                        delay_ifg
                                    )
                                )
                            )

        # open time constraint
        for flow in self.flows:
            path = flow.path
            for link_id in path:
                link = self.links_dict[link_id]
                hold_time = link.transmission_time(12 + 1522 + 8)  # inter-frame gap + mtu + preamble
                self.constraints_set.append(
                    z3.If(
                        self.z3_variables_flow_link[flow][link]['gc'],
                        z3.And(
                            self.z3_variables_flow_link[flow][link]['t4_min'] ==
                            self.z3_variables_flow_link[flow][link]['t4_max'],
                            self.z3_variables_flow_link[flow][link]['t4_min'] >=
                            self.z3_variables_flow_link[flow][link]['t3_max']
                        ),
                        z3.And(
                            self.z3_variables_flow_link[flow][link]['t4_min'] ==
                            self.z3_variables_flow_link[flow][link]['t3_min'],
                            self.z3_variables_flow_link[flow][link]['t4_max'] ==
                            self.z3_variables_flow_link[flow][link]['t3_max'] + hold_time
                        )
                    )
                )

                self.constraints_set.append(
                    z3.If(
                        self.z3_variables_flow_link[flow][link]['gc'],
                        self.z3_variables_flow_link[flow][link]['tb'] ==
                        self.z3_variables_flow_link[flow][link]['t4_min'],
                        self.z3_variables_flow_link[flow][link]['tb'] ==
                        self.z3_variables_flow_link[flow][link]['t3_min']
                    )
                )

                # rr constraint
                self.constraints_set.append(
                    z3.If(
                        self.z3_variables_flow_link[flow][link]['gc'],
                        flow.period * self.z3_variables_flow_link[flow][link]['rr'] ==
                        self.z3_variables_links[link]['gcl_cycle'],
                        self.z3_variables_flow_link[flow][link]['rr'] == 0
                    )
                )

        # GCL capacity limit
        for link, link_variables in self.z3_variables_links.items():
            link_id = link.link_id
            flows = list(filter(lambda f: link_id in f.path, self.flows))
            rrs = [self.z3_variables_flow_link[f][link]['rr'] for f in flows]
            if len(rrs) > 0:
                self.constraints_set.append(
                    self.z3_variables_links[link]['gcl_cycle'] > 0
                )
                self.constraints_set.append(
                    z3.Sum(rrs) * 2 ==
                    self.z3_variables_links[link]['gcl_length']
                )
                self.constraints_set.append(
                    self.z3_variables_links[link]['gcl_length'] <= link.max_gcl_length
                )

    def _solve_constraints(self) -> bool:
        solver = z3.Solver()
        for constraint in self.constraints_set:
            solver.add(constraint)

        is_sat = solver.check()

        if is_sat == z3.sat:
            self.model = solver.model()
            logging.info(f"Successfully scheduled.")
            return True
        elif is_sat == z3.unsat:
            logging.error("z3 fail to find a valid solution.")
        elif is_sat == z3.unknown:
            logging.error(f"z3 unknown: {solver.reason_unknown()}")
        else:
            raise NotImplementedError
        return False

    def save_results(self, filename):
        assert self.model is not None, "Not yet scheduled."
        res = []
        for i, flow in enumerate(self.flows):
            path = flow.path
            res += [f"{i}. {str(flow)}\n"]
            for link_id in path:
                link = self.links_dict[link_id]
                variables = self.z3_variables_flow_link[flow][link]
                res += [f"\t{link_id}, Operation({self.model[variables['t3_min']]}, "
                        f"{self.model[variables['t4_min']] if self.model[variables['gc']] else None}, "
                        f"{self.model[variables['t5_max']]})("
                        f"{self.model[variables['t4_min']]}, {self.model[variables['t4_max']]})\n"]

        res += ["\nSMT variables: \n"] + \
               [f"\t{declare.name()}: {self.model[declare]}\n" for declare in self.model.decls()]
        with open(filename, 'w') as f:
            f.writelines(res)


class NoWaitSmtScheduler(SmtScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _construct_constraints(self):
        super()._construct_constraints()

        for flow in self.flows:
            path = flow.path
            for link_id in path:
                link = self.links_dict[link_id]
                hold_time = link.transmission_time(12 + 1522 + 8)  # inter-frame gap + mtu + preamble
                self.constraints_set.append(
                    z3.If(
                        self.z3_variables_flow_link[flow][link]['gc'],
                        # no-wait, although enable gating, t4_min should equal to t3_max
                        z3.And(
                            self.z3_variables_flow_link[flow][link]['t4_min'] ==
                            self.z3_variables_flow_link[flow][link]['t4_max'],
                            self.z3_variables_flow_link[flow][link]['t4_min'] ==
                            self.z3_variables_flow_link[flow][link]['t3_max']
                        ),
                        z3.And(
                            self.z3_variables_flow_link[flow][link]['t4_min'] ==
                            self.z3_variables_flow_link[flow][link]['t3_min'],
                            self.z3_variables_flow_link[flow][link]['t4_max'] ==
                            self.z3_variables_flow_link[flow][link]['t3_max'] + hold_time
                        )
                    )
                )
