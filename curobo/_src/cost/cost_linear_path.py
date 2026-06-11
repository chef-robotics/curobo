# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cost that guides the tool along a straight start->goal Cartesian path.

For each non-terminal trajectory step the cost penalizes the perpendicular
deviation of the tool position from the segment ``line_start -> line_end`` and
the deviation of the tool orientation from ``slerp(quat_start, quat_goal, t)``
where ``t`` is the arc-length fraction of the tool's position along the segment.

The endpoints are stored as in-place buffers so they (and the cost weight and
deadband tolerance) can be updated between solves without changing tensor shapes
-- this keeps the cost compatible with CUDA-graph capture. Only single-problem
solves are supported (``num_goalset == 1``); the line is defined per tool frame,
broadcast across the whole optimization batch and horizon.

Cost vs. constraint
-------------------
The same class serves two roles depending on its ``tolerance``:

* Soft guiding **cost** (``tolerance == 0``): a plain quadratic that nudges the
  path toward the line without forcing it. Registered under ``cost_cfg`` as
  ``linear_path``.
* Tolerance-gated **constraint** (``tolerance > 0``): a one-sided hinge that is
  exactly zero within the tolerance tube and positive beyond it. Registered
  under ``constraint_cfg`` as ``linear_path_constraint``; cuRobo's feasibility
  gate marks a trajectory infeasible when the summed constraint value exceeds
  zero, so deviation beyond the tolerance fails the solve (and drives retries /
  seed reselection toward a linear-feasible solution). Register the same
  component in both ``trajopt/lbfgs_bspline_trajopt.yml`` and
  ``metrics_base.yml`` so the optimizer and success check agree; keep
  tolerances in sync (matching weights is recommended).

The two are independent instances with their own weights/tolerances, addressed
by their distinct registered names.

Enabling/disabling via weight (not ``enable_cost_component``)
------------------------------------------------------------
This cost is intended to stay registered and enabled for the lifetime of the
solver, and is turned on/off by setting its weight (and endpoints) -- a zero
weight (or a degenerate, zero-length segment) makes it inert. We deliberately do
NOT use ``RobotCostManager.enable_cost_component`` /
``disable_cost_component`` to toggle it: those flip whether the kernel is
launched at all in ``compute_costs``, which changes the set of operations in the
optimizer's hot loop. cuRobo captures that loop into a CUDA graph (during
``MotionPlanner.warmup`` and the first solves); toggling the component
afterwards would not be reflected in the already-captured graph (and would
otherwise force an expensive graph re-capture). Updating the weight/endpoints
in place changes only buffer *values* the captured kernel already reads, so it
is CUDA-graph safe. Likewise, setting the orientation weight to 0 disables only
the orientation term, leaving the perpendicular position term active.
"""

from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Optional, Tuple

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.wp_linear_path import LinearPathDistance
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_linear_path_cfg import LinearPathCostCfg


class LinearPathCost(BaseCost):
    """Soft cost penalizing deviation from a linear tool-space path."""

    def __init__(self, config: "LinearPathCostCfg"):
        """Initialize the cost and allocate per-link endpoint buffers.

        Args:
            config: Linear-path cost configuration. ``config.tool_frames`` must
                be populated (the cost manager calls ``set_tool_frames`` first).
        """
        self.config: "LinearPathCostCfg" = config
        if config.tool_frames is None:
            log_and_raise("LinearPathCost requires tool_frames to be set")
        self.tool_frames = list(config.tool_frames)
        self.num_links = len(self.tool_frames)

        super().__init__(config)

        device = self.device_cfg.device
        # Degenerate (zero-length) segments by default: the cost is inert until
        # real endpoints are provided via :meth:`update_linear_path`.
        self._line_start = torch.zeros(
            (self.num_links, 3), dtype=torch.float32, device=device
        )
        self._line_end = torch.zeros(
            (self.num_links, 3), dtype=torch.float32, device=device
        )
        # Identity quaternions in (w, x, y, z) order.
        self._quat_start = torch.zeros(
            (self.num_links, 4), dtype=torch.float32, device=device
        )
        self._quat_goal = torch.zeros(
            (self.num_links, 4), dtype=torch.float32, device=device
        )
        self._quat_start[:, 0] = 1.0
        self._quat_goal[:, 0] = 1.0
        # Deadband tolerances ``[position_m, orientation_rad]``. Zero -> plain
        # quadratic terms (soft guiding cost); non-zero -> hinge (used when this
        # instance is registered as a feasibility constraint).
        self._tolerance = torch.tensor(
            [float(config.position_tolerance), float(config.orientation_tolerance)],
            dtype=torch.float32,
            device=device,
        )
        if not config.active_at_init:
            self.clear_linear_path()

    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs) -> None:
        if batch_size != self._batch_size or horizon != self._horizon:
            device = self.device_cfg.device
            self._out_distance = torch.zeros(
                (batch_size, horizon, 2 * self.num_links),
                dtype=torch.float32,
                device=device,
            )
            self._out_position_gradient = torch.zeros(
                (batch_size, horizon, self.num_links, 3),
                dtype=torch.float32,
                device=device,
            )
            self._out_rotation_gradient = torch.zeros(
                (batch_size, horizon, self.num_links, 4),
                dtype=torch.float32,
                device=device,
            )
        super().setup_batch_tensors(batch_size, horizon)

    def forward(
        self,
        current_tool_poses: ToolPose,
        goal_tool_poses: Optional[GoalToolPose] = None,
        idxs_goal: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the linear-path cost for a tool-pose trajectory.

        Args:
            current_tool_poses: Current tool poses with ``position``
                ``(b, h, num_links, 3)`` and ``quaternion``
                ``(b, h, num_links, 4)`` (w, x, y, z).
            goal_tool_poses: Optional goal poses; only used to assert that a
                single goal is being solved (``num_goalset == 1``).
            idxs_goal: Unused; accepted for signature parity with other costs.

        Returns:
            Cost tensor of shape ``(b, h, num_links * 2)``.
        """
        if current_tool_poses is None:
            log_and_raise("current_tool_poses must be provided")
        if goal_tool_poses is not None and goal_tool_poses.num_goalset > 1:
            log_and_raise(
                "LinearPathCost only supports single-goal solves (num_goalset == 1)"
            )
        if current_tool_poses.tool_frames != self.tool_frames:
            log_and_raise(
                "current_tool_poses tool frames do not match LinearPathCost tool frames"
            )

        cost = LinearPathDistance.apply(
            current_tool_poses.position,
            current_tool_poses.quaternion,
            self._line_start,
            self._line_end,
            self._quat_start,
            self._quat_goal,
            self._weight,
            self._tolerance,
            self._out_distance,
            self._out_position_gradient,
            self._out_rotation_gradient,
            self.config.use_grad_input,
        )
        return cost

    def _restore_weight_from_config(self) -> None:
        """Copy the task-YAML weight into the live buffer (CUDA-graph safe)."""
        self._weight.copy_(self.config.weight)

    def _restore_tolerance_from_config(self) -> None:
        """Copy the task-YAML deadband into the live buffer."""
        self._tolerance.copy_(
            torch.tensor(
                [
                    float(self.config.position_tolerance),
                    float(self.config.orientation_tolerance),
                ],
                dtype=self._tolerance.dtype,
                device=self._tolerance.device,
            )
        )

    def update_linear_path(
        self,
        line_start: torch.Tensor,
        line_end: torch.Tensor,
        quat_start: torch.Tensor,
        quat_goal: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        tolerance: Optional[torch.Tensor] = None,
    ) -> None:
        """Set linear-path endpoints; restore weight/tolerance from config when omitted."""
        self._line_start.copy_(self._as_link_tensor(line_start, 3, "line_start"))
        self._line_end.copy_(self._as_link_tensor(line_end, 3, "line_end"))
        self._quat_start.copy_(self._as_link_tensor(quat_start, 4, "quat_start"))
        self._quat_goal.copy_(self._as_link_tensor(quat_goal, 4, "quat_goal"))
        if weight is not None:
            weight_t = torch.as_tensor(
                weight, dtype=self._weight.dtype, device=self._weight.device
            ).reshape(-1)
            if weight_t.numel() != 2:
                log_and_raise("LinearPathCost weight must have 2 elements")
            self._weight.copy_(weight_t)
        else:
            self._restore_weight_from_config()
        if tolerance is not None:
            tol_t = torch.as_tensor(
                tolerance, dtype=self._tolerance.dtype, device=self._tolerance.device
            ).reshape(-1)
            if tol_t.numel() != 2:
                log_and_raise("LinearPathCost tolerance must have 2 elements")
            self._tolerance.copy_(tol_t)
        else:
            self._restore_tolerance_from_config()

    def clear_linear_path(self) -> None:
        """Disable the linear-path term without unregistering it (CUDA-graph safe).

        Collapses the segment to zero length and zeroes the live weight buffer.
        The kernel stays registered in the captured optimizer loop; a zero
        weight makes the term inert. Call :meth:`update_linear_path` (without
        overrides) to re-enable using the task-YAML weight/tolerance.
        """
        self._line_end.copy_(self._line_start)
        self._weight.zero_()

    def _as_link_tensor(
        self, value: torch.Tensor, dim: int, name: str
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            value, dtype=torch.float32, device=self.device_cfg.device
        ).reshape(-1, dim)
        if tensor.shape[0] == 1 and self.num_links > 1:
            tensor = tensor.expand(self.num_links, dim)
        if tensor.shape != (self.num_links, dim):
            log_and_raise(
                f"{name} must broadcast to shape ({self.num_links}, {dim}), "
                f"got {tuple(tensor.shape)}"
            )
        return tensor
