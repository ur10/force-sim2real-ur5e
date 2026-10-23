"""The contact-probe protocol of plan step 1 / step 5, shared by the simulated run
(run_contact_probe.py) and, later, the real one, so the two produce comparable traces.

Each condition: hold the ring above the rod tip with a known lateral offset (along base x) and
tilt (about base y), descend along the rod axis at a fixed speed with compliance only along the
axis (damping control: a blocked ring pushes back with at most damping x speed), hold, retract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product


@dataclass
class ProbeCondition:
    offset_mm: float
    tilt_deg: float
    speed_mm_s: float = 5.0
    start_gap_mm: float = 8.0       # ring bottom above the rod tip at the start
    target_depth_mm: float = 15.0   # how far the reference goes below the tip
    hold_s: float = 0.5
    abort_force_n: float = 40.0     # stop and retract above this (simulation stand-in for the lab limit)


@dataclass
class ProbeProtocol:
    offsets_mm: tuple[float, ...] = (0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0)
    tilts_deg: tuple[float, ...] = (0.0, 3.0, 6.0)
    speed_mm_s: float = 5.0
    repeats: int = 1
    conditions: list[ProbeCondition] = field(default_factory=list)

    def expand(self) -> list[ProbeCondition]:
        if self.conditions:
            return self.conditions
        return [ProbeCondition(o, t, self.speed_mm_s) for o, t in product(self.offsets_mm, self.tilts_deg)
                for _ in range(self.repeats)]
