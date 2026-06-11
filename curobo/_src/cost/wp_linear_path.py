# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernel and autograd function for the linear TCP path cost.

The kernel penalizes, at every non-terminal trajectory step, the perpendicular
deviation of the tool position from the straight segment ``line_start ->
line_end`` and the deviation of the tool orientation from a slerp interpolation
``slerp(quat_start, quat_goal, t)`` where ``t`` is the arc-length fraction of the
tool's position along the segment (the clamped position projection). Tying the
orientation target to spatial progress keeps orientation synchronized with how
far the tool has travelled down the line; ``t`` is treated as detached so the
rotation gradient flows only through the current quaternion.

A per-term ``tolerance`` (``[position_m, orientation_rad]``) introduces a
deadband: deviation within the tolerance contributes zero cost and zero
gradient, and only the excess is penalized (a one-sided hinge). With a zero
tolerance this reduces exactly to the plain quadratic terms (``0.5*w*||perp||^2``
and ``w*angle^2`` with gradient ``w*perp`` / the usual rotation gradient). The
deadband is what lets the same term act as a tolerance-gated feasibility
*constraint*: cuRobo marks a trajectory feasible when the summed constraint
value is ``<= 0``, so a hinge that is exactly zero inside the tolerance tube
means "within tolerance == feasible".

This mirrors the structure of :mod:`curobo._src.cost.wp_tool_pose`; the
orientation error and quaternion-rate gradient reuse the shared warp functions
defined there.
"""

# Standard Library
from typing import Optional

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.cost.wp_tool_pose import (
    compute_rotation_error,
    convert_angular_velocity_to_quaternion_rate,
)
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream


@wp.func
def slerp_quaternion(q0: wp.quat, q1: wp.quat, t: wp.float32):
    """Spherical linear interpolation between two quaternions.

    Args:
        q0: Start quaternion in (x, y, z, w) order.
        q1: End quaternion in (x, y, z, w) order.
        t: Interpolation parameter, assumed already clamped to [0, 1].

    Returns:
        Interpolated unit quaternion in (x, y, z, w) order.
    """
    d = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]

    # Take the shorter arc.
    q1b = q1
    if d < 0.0:
        q1b = wp.quat(-q1[0], -q1[1], -q1[2], -q1[3])
        d = -d

    if d > 0.9995:
        # Quaternions are nearly parallel: normalized linear interpolation.
        r = wp.quat(
            q0[0] + t * (q1b[0] - q0[0]),
            q0[1] + t * (q1b[1] - q0[1]),
            q0[2] + t * (q1b[2] - q0[2]),
            q0[3] + t * (q1b[3] - q0[3]),
        )
        n = wp.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3])
        if n < 1e-9:
            return wp.quat(0.0, 0.0, 0.0, 1.0)
        return wp.quat(r[0] / n, r[1] / n, r[2] / n, r[3] / n)

    theta_0 = wp.acos(d)
    theta = theta_0 * t
    sin_theta = wp.sin(theta)
    sin_theta_0 = wp.sin(theta_0)

    s1 = sin_theta / sin_theta_0
    s0 = wp.cos(theta) - d * s1

    return wp.quat(
        s0 * q0[0] + s1 * q1b[0],
        s0 * q0[1] + s1 * q1b[1],
        s0 * q0[2] + s1 * q1b[2],
        s0 * q0[3] + s1 * q1b[3],
    )


@wp.kernel
def linear_path_distance(
    current_position: wp.array(dtype=wp.vec3),  # [batch * horizon * num_links]
    current_quat: wp.array(dtype=wp.vec4),  # [batch * horizon * num_links] (w,x,y,z)
    line_start: wp.array(dtype=wp.vec3),  # [num_links]
    line_end: wp.array(dtype=wp.vec3),  # [num_links]
    quat_start: wp.array(dtype=wp.vec4),  # [num_links] (w,x,y,z)
    quat_goal: wp.array(dtype=wp.vec4),  # [num_links] (w,x,y,z)
    position_orientation_weight: wp.array(dtype=wp.float32),  # [2]
    position_orientation_tolerance: wp.array(dtype=wp.float32),  # [2]
    out_distance: wp.array(dtype=wp.float32),  # [batch * horizon * num_links * 2]
    out_position_gradient: wp.array(dtype=wp.vec3),  # [batch * horizon * num_links]
    out_rotation_gradient: wp.array(dtype=wp.vec4),  # [batch * horizon * num_links]
    batch_size: wp.int32,
    horizon: wp.int32,
    num_links: wp.int32,
):
    tid = wp.tid()
    if tid >= batch_size * horizon * num_links:
        return

    b_idx = tid / (horizon * num_links)
    h_idx = (tid - b_idx * horizon * num_links) / num_links
    link_idx = tid - b_idx * horizon * num_links - h_idx * num_links

    # Default to zero so disabled / terminal / degenerate threads contribute nothing.
    out_distance[2 * tid] = 0.0
    out_distance[2 * tid + 1] = 0.0
    out_position_gradient[tid] = wp.vec3(0.0, 0.0, 0.0)
    out_rotation_gradient[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)

    # The terminal step is anchored by the tool-pose cost; the first step is
    # anchored by the start state. Only constrain interior / non-terminal steps.
    if horizon > 1 and h_idx >= horizon - 1:
        return

    position_weight = position_orientation_weight[0]
    rotation_weight = position_orientation_weight[1]
    position_tol = position_orientation_tolerance[0]
    orientation_tol = position_orientation_tolerance[1]

    s = line_start[link_idx]
    e = line_end[link_idx]
    u = e - s
    seg_sq = wp.dot(u, u)
    if seg_sq < 1.0e-9:
        # Degenerate segment (start == goal): no linear-path constraint.
        return

    c_pos = current_position[tid]
    c_quat = current_quat[tid]  # w, x, y, z
    c_quaternion = wp.quaternion(c_quat[1], c_quat[2], c_quat[3], c_quat[0])  # x, y, z, w

    # Perpendicular deviation from the infinite line through (s, e).
    v = c_pos - s
    t = wp.dot(v, u) / seg_sq
    perp = v - t * u

    # Position deadband (one-sided hinge): penalize only the perpendicular
    # distance beyond ``position_tol``. With tol == 0 this is exactly
    # 0.5*w*||perp||^2 with gradient w*perp (the orthogonal projector is constant
    # in p, so d/dp ||perp|| = perp/||perp||).
    perp_dist = wp.length(perp)
    position_distance = float(0.0)
    position_gradient = wp.vec3(0.0, 0.0, 0.0)
    if perp_dist > position_tol and perp_dist > 1.0e-9:
        perp_over = perp_dist - position_tol
        position_distance = 0.5 * position_weight * perp_over * perp_over
        position_gradient = (position_weight * perp_over / perp_dist) * perp

    # Orientation target: slerp by spatial progress ALONG the segment, i.e. the
    # arc-length fraction given by the position projection ``t`` (clamped to the
    # segment). This ties the orientation interpolation to where the tool is on
    # the line, so the tool is "equally rotated through" as it advances -- not to
    # the (non-uniformly timed) horizon index. The dependence of the target on
    # the position is treated as detached: the rotation gradient below flows only
    # through ``current_quat`` (the perpendicular position term owns the position
    # gradient), which keeps the gradient well-defined and the solve stable.
    t_clamped = wp.clamp(t, 0.0, 1.0)
    qs = quat_start[link_idx]  # w, x, y, z
    qg = quat_goal[link_idx]
    qs_xyzw = wp.quaternion(qs[1], qs[2], qs[3], qs[0])
    qg_xyzw = wp.quaternion(qg[1], qg[2], qg[3], qg[0])
    q_target = slerp_quaternion(qs_xyzw, qg_xyzw, t_clamped)

    rotation_dof_weight = wp.vec3(1.0, 1.0, 1.0)
    angular_distance, gradient_as_angular_velocity, angle = compute_rotation_error(
        c_quaternion,
        q_target,
        rotation_dof_weight,
        rotation_weight,
        0.0,
        0,
    )

    # Orientation deadband (one-sided hinge in the geometric angle): penalize
    # only rotation error beyond ``orientation_tol`` radians. ``compute_rotation_error``
    # (axis-angle) returns angular_distance = w*angle^2 and a gradient of
    # magnitude proportional to ``angle`` along the rotation axis; rescaling by
    # ``(angle - tol)/angle`` turns these into w*(angle - tol)^2 and its
    # gradient. With tol == 0 the scale is 1 and both are left unchanged.
    if rotation_weight > 0.0 and angle > orientation_tol:
        angle_over = angle - orientation_tol
        angular_distance = rotation_weight * angle_over * angle_over
        gradient_as_angular_velocity = (angle_over / angle) * gradient_as_angular_velocity
    else:
        angular_distance = 0.0
        gradient_as_angular_velocity = wp.vec3(0.0, 0.0, 0.0)

    quaternion_rate_gradient = convert_angular_velocity_to_quaternion_rate(
        gradient_as_angular_velocity, c_quaternion
    )

    out_distance[2 * tid] = position_distance
    out_distance[2 * tid + 1] = angular_distance
    out_position_gradient[tid] = position_gradient
    out_rotation_gradient[tid] = wp.vec4(
        quaternion_rate_gradient[3],  # w
        quaternion_rate_gradient[0],  # x
        quaternion_rate_gradient[1],  # y
        quaternion_rate_gradient[2],  # z
    )


class LinearPathDistance(torch.autograd.Function):
    """Autograd bridge for :func:`linear_path_distance`.

    Only ``current_position`` and ``current_quat`` receive gradients; the line
    endpoints, target quaternions, weights, and tolerances are treated as
    constants.
    """

    @staticmethod
    def forward(
        ctx,
        current_position: torch.Tensor,
        current_quat: torch.Tensor,
        line_start: torch.Tensor,
        line_end: torch.Tensor,
        quat_start: torch.Tensor,
        quat_goal: torch.Tensor,
        position_orientation_weight: torch.Tensor,
        position_orientation_tolerance: torch.Tensor,
        out_distance: torch.Tensor,
        out_position_gradient: torch.Tensor,
        out_rotation_gradient: torch.Tensor,
        use_grad_input: bool,
    ):
        """Compute the linear-path distance and gradients.

        Args:
            ctx: Autograd context.
            current_position: Shape ``(b, h, num_links, 3)``.
            current_quat: Shape ``(b, h, num_links, 4)`` in (w, x, y, z) order.
            line_start: Shape ``(num_links, 3)``. Segment start in base frame.
            line_end: Shape ``(num_links, 3)``. Segment end in base frame.
            quat_start: Shape ``(num_links, 4)`` in (w, x, y, z) order.
            quat_goal: Shape ``(num_links, 4)`` in (w, x, y, z) order.
            position_orientation_weight: Shape ``(2,)``: position weight then
                orientation weight.
            position_orientation_tolerance: Shape ``(2,)``: position tolerance
                (metres) then orientation tolerance (radians). Deviation within
                the tolerance contributes zero cost/gradient (one-sided hinge).
            out_distance: Pre-allocated output, shape ``(b, h, num_links * 2)``.
            out_position_gradient: Pre-allocated, shape ``(b, h, num_links, 3)``.
            out_rotation_gradient: Pre-allocated, shape ``(b, h, num_links, 4)``.
            use_grad_input: If True, multiply stored gradients by the incoming
                upstream gradient in the backward pass.

        Returns:
            ``out_distance`` of shape ``(b, h, num_links * 2)``.
        """
        ctx.set_materialize_grads(False)
        if current_position.ndim != 4:
            log_and_raise("current_position must be a 4D tensor")
        if current_quat.ndim != 4:
            log_and_raise("current_quat must be a 4D tensor")

        b, h, num_links, _ = current_position.shape
        if current_position.shape != (b, h, num_links, 3):
            log_and_raise("current_position must have shape (b, h, num_links, 3)")
        if current_quat.shape != (b, h, num_links, 4):
            log_and_raise("current_quat must have shape (b, h, num_links, 4)")
        if line_start.shape != (num_links, 3) or line_end.shape != (num_links, 3):
            log_and_raise("line_start/line_end must have shape (num_links, 3)")
        if quat_start.shape != (num_links, 4) or quat_goal.shape != (num_links, 4):
            log_and_raise("quat_start/quat_goal must have shape (num_links, 4)")
        if position_orientation_weight.shape != (2,):
            log_and_raise("position_orientation_weight must have shape (2,)")
        if position_orientation_tolerance.shape != (2,):
            log_and_raise("position_orientation_tolerance must have shape (2,)")
        if out_distance.shape != (b, h, num_links * 2):
            log_and_raise("out_distance must have shape (b, h, num_links*2)")
        if out_position_gradient.shape != (b, h, num_links, 3):
            log_and_raise("out_position_gradient must have shape (b, h, num_links, 3)")
        if out_rotation_gradient.shape != (b, h, num_links, 4):
            log_and_raise("out_rotation_gradient must have shape (b, h, num_links, 4)")

        ctx.use_grad_input = use_grad_input
        wp_device, wp_stream = get_warp_device_stream(current_position)

        wp.launch(
            kernel=linear_path_distance,
            dim=b * h * num_links,
            inputs=[
                wp.from_torch(current_position.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(current_quat.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(line_start.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(line_end.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(quat_start.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(quat_goal.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(position_orientation_weight.view(-1), dtype=wp.float32),
                wp.from_torch(position_orientation_tolerance.view(-1), dtype=wp.float32),
                wp.from_torch(out_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_position_gradient.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(out_rotation_gradient.view(-1, 4), dtype=wp.vec4),
                b,
                h,
                num_links,
            ],
            device=wp_device,
            stream=wp_stream,
            adjoint=False,
        )

        ctx.mark_non_differentiable(
            line_start,
            line_end,
            quat_start,
            quat_goal,
            position_orientation_weight,
            position_orientation_tolerance,
        )
        ctx.save_for_backward(out_position_gradient, out_rotation_gradient)
        return out_distance

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_distance: Optional[torch.Tensor]):
        use_grad_input = ctx.use_grad_input
        pos_grad = None
        quat_grad = None
        if grad_distance is not None:
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                out_position_gradient, out_rotation_gradient = ctx.saved_tensors
            if ctx.needs_input_grad[0]:
                if use_grad_input:
                    grad_pos = grad_distance[:, :, 0::2].unsqueeze(-1)
                    pos_grad = out_position_gradient * grad_pos
                else:
                    pos_grad = out_position_gradient
            if ctx.needs_input_grad[1]:
                if use_grad_input:
                    grad_ori = grad_distance[:, :, 1::2].unsqueeze(-1)
                    quat_grad = out_rotation_gradient * grad_ori
                else:
                    quat_grad = out_rotation_gradient

        return (
            pos_grad,  # current_position
            quat_grad,  # current_quat
            None,  # line_start
            None,  # line_end
            None,  # quat_start
            None,  # quat_goal
            None,  # position_orientation_weight
            None,  # position_orientation_tolerance
            None,  # out_distance
            None,  # out_position_gradient
            None,  # out_rotation_gradient
            None,  # use_grad_input
        )
