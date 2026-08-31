"""
The hardware throw session: cold rig -> calibrated -> N real throws -> a model
update on that data.

Spec: docs/superpowers/specs/2026-08-31-hardware-session-design.md

Stage 0 runs BEFORE the camera thread starts, because calibration needs colour
at 1920x1080 while tracking needs IR at 848x480/90fps. Sequential, so there is
no stream reconfiguration mid-session and start_of_day.py is reused exactly as
it is, opening and closing the camera itself.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional

from hardware_learning import scale_allowed


class Stage(enum.Enum):
    COLD = "cold"
    BLOCKED = "blocked"
    CALIBRATED = "calibrated"
    READY = "ready"
    MODEL_UPDATED = "model_updated"


@dataclass
class SessionState:
    """
    Every safety rule the session enforces lives here, and nowhere else, so it
    can be tested without a window or an arm.
    """
    min_throws_for_update: int = 5
    stage: Stage = Stage.COLD
    blocked_reason: str = ""
    confirmed: bool = False
    throws: List[dict] = field(default_factory=list)
    model_updated: bool = False

    @property
    def n_throws(self):
        return len(self.throws)

    @property
    def n_measured(self):
        return sum(1 for t in self.throws if t.get("landing_xy") is not None)

    @property
    def logged_scales(self):
        return [t["speed_scale"] for t in self.throws
                if t.get("landing_xy") is not None]

    def record_startup(self, go, failures):
        self.stage = Stage.CALIBRATED if go else Stage.BLOCKED
        self.blocked_reason = "" if go else "; ".join(failures)

    def camera_ready(self):
        if self.stage is Stage.CALIBRATED:
            self.stage = Stage.READY

    def can_throw(self):
        return self.stage in (Stage.READY, Stage.MODEL_UPDATED)

    def check_scale(self, requested):
        return scale_allowed(requested, self.logged_scales)

    def record_throw(self, record):
        self.throws.append(record)
        self.confirmed = False        # re-affirm every single time

    def can_update_model(self):
        return self.n_measured >= self.min_throws_for_update

    def record_model_update(self):
        self.model_updated = True
        self.stage = Stage.MODEL_UPDATED

    def can_reoptimize_policy(self):
        return self.model_updated
