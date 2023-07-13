import logging
import networkx as nx
import z3

from src.network.net import Flow


class SmtScheduler:
    def __init__(self, graph: nx.Graph, flows: list[Flow]):
        self.constraints_set = []
        pass

    def schedule(self):
        self._construct_constraints()
        self._solve_constraints()
        self.save_results()

    def _construct_constraints(self):
        x = z3.Int('x')
        y = z3.Int('y')

        self.constraints_set.append(
            z3.And(x > 10, y == x + 2)
        )

    def _solve_constraints(self):
        solver = z3.Solver()
        for constraint in self.constraints_set:
            solver.add(constraint)

        is_sat = solver.check()

        if is_sat == z3.sat:
            model = solver.model()
            solution = []
            for declare in model.decls():
                name = declare.name()
                value = model[declare]
                solution.append({'name': name, 'value': value})
            logging.debug(solution)
        elif is_sat == z3.unsat:
            logging.error("z3 fail to find a valid solution.")
        elif is_sat == z3.unknown:
            logging.error(f"z3 unknown: {solver.reason_unknown()}")
        else:
            raise NotImplementedError

    def save_results(self):
        pass
