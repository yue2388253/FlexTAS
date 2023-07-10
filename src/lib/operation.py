import copy
from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class Operation:
    start_time: int
    gating_time: Optional[int]
    latest_time: int  # must equal to gating time if enable gating
    end_time: int

    def __post_init__(self):
        if self.gating_time is not None:
            assert self.start_time <= self.gating_time == self.latest_time < self.end_time, "Invalid Operation"
        else:
            assert self.start_time <= self.latest_time < self.end_time, "Invalid Operation"

        self.earliest_time = self.start_time if self.gating_time is None else self.gating_time

    def add(self, other: int):
        assert isinstance(other, int), "Operation can only add an integer"
        self.start_time += other
        if self.gating_time is not None:
            self.gating_time += other
        self.latest_time += other
        self.end_time += other
        self.earliest_time += other
        return self

    def __repr__(self):
        return f"Operation({self.start_time, self.gating_time, self.end_time})"


# add tests
def check_operation_isolation(operation1: tuple[Operation, int],
                              operation2: tuple[Operation, int],
                              safe_distance: int) -> Optional[int]:
    """

    :param operation1:
    :param operation2:
    :param safe_distance:
    :return: None if isolation constraint is satisfied,
             otherwise, it returns the offset that `opertion1` should add.
             Notice that the adding the returned offset might make `operation` out of period.
    """
    operation1, period1 = operation1
    operation2, period2 = operation2

    assert (operation1.start_time >= 0) and (operation1.end_time <= period1)
    assert (operation2.start_time >= 0) and (operation2.end_time <= period2)

    hyper_period = math.lcm(period1, period2)
    alpha = hyper_period // period1
    beta = hyper_period // period2

    operation_lhs = copy.deepcopy(operation1)

    for i in range(alpha):
        operation_rhs = copy.deepcopy(operation2)
        for j in range(beta):
            if (operation_lhs.start_time - safe_distance <= operation_rhs.start_time < operation_lhs.end_time + safe_distance) or \
                    (operation_rhs.start_time - safe_distance <= operation_lhs.start_time < operation_rhs.end_time):
                return operation_rhs.end_time - operation_lhs.start_time + safe_distance
            operation_rhs.add(period2)
        operation_lhs.add(period1)
    return None
