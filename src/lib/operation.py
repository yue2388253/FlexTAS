from dataclasses import dataclass
from typing import Optional


@dataclass
class Operation:
    start_time: int
    gating_time: Optional[int]
    latest_time: int  # must equal to gating time if enable gating
    end_time: int

    def __post_init__(self):
        if self.gating_time is not None:
            assert self.start_time < self.gating_time == self.latest_time < self.end_time, "Invalid Operation"
        else:
            assert self.start_time < self.latest_time < self.end_time, "Invalid Operation"

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
