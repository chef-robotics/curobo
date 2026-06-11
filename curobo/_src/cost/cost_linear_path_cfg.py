# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the linear TCP path cost."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional, Type

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_linear_path import LinearPathCost


@dataclass
class LinearPathCostCfg(BaseCostCfg):
    """Configuration for :class:`LinearPathCost`.

    The cost penalizes per-step deviation of each tool frame from the straight
    segment between a start and a goal pose. ``weight`` is a length-2 vector:
    the position weight followed by the orientation weight.

    A non-zero ``position_tolerance`` / ``orientation_tolerance`` introduces a
    deadband (one-sided hinge): deviation within the tolerance contributes zero
    cost and zero gradient. This is what lets a second instance of this cost act
    as a tolerance-gated *feasibility constraint* (registered under
    ``constraint_cfg``): cuRobo treats a trajectory as feasible when the summed
    constraint value is ``<= 0``, so a hinge that is exactly zero inside the
    tolerance tube means "within tolerance == feasible". The soft guiding cost
    leaves both tolerances at ``0.0`` (a plain quadratic).
    """

    #: Concrete cost class instantiated from this configuration.
    class_type: Type[LinearPathCost] = LinearPathCost

    #: List of tool frame (link) names this cost applies to. Set by the cost
    #: manager from the robot model via :meth:`set_tool_frames`.
    tool_frames: Optional[List[str]] = None

    #: Position deadband in metres. Perpendicular deviation within this distance
    #: of the straight segment contributes no cost/gradient. ``0.0`` -> plain
    #: quadratic position term.
    position_tolerance: float = 0.0

    #: Orientation deadband in radians. Rotation error within this angle of the
    #: slerp target contributes no cost/gradient. ``0.0`` -> plain quadratic
    #: orientation term.
    orientation_tolerance: float = 0.0

    #: When ``False`` (default), the live weight buffer is zeroed at construction
    #: so the term is inert until :meth:`LinearPathCost.update_linear_path` is
    #: called. YAML ``weight`` / tolerances remain in ``self.config`` as the
    #: source of truth for the next activation. When ``True``, the configured
    #: weight is copied into the live buffer at init (endpoints still default to
    #: a degenerate segment until updated).
    active_at_init: bool = False

    def set_tool_frames(self, tool_frames: List[str]) -> None:
        """Record the tool frames the cost applies to.

        Args:
            tool_frames: Ordered list of link names.
        """
        self.tool_frames = list(tool_frames)

    @property
    def num_links(self) -> int:
        """Number of tool frames."""
        return 0 if self.tool_frames is None else len(self.tool_frames)

    def clone(self) -> "LinearPathCostCfg":
        """Create a deep copy of this configuration."""
        return LinearPathCostCfg(
            weight=self.weight.clone(),
            device_cfg=self.device_cfg,
            convert_to_binary=self.convert_to_binary,
            use_grad_input=self.use_grad_input,
            tool_frames=None if self.tool_frames is None else list(self.tool_frames),
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            active_at_init=self.active_at_init,
        )
