#!/usr/bin/env python3

"""Shared runtime helpers for non-generative traffic simulation."""

import numpy as np


def get_bool_param(rospy, name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def clamp_vehicle_state(position, speed, acceleration, dt, speed_limit, acc_min, acc_max):
    """Integrate one vehicle while enforcing physical limits."""
    dt = max(float(dt), 0.0)
    acceleration = float(np.clip(acceleration, acc_min, acc_max))
    next_speed = float(np.clip(speed + acceleration * dt, 0.0, speed_limit))
    next_position = float(position + speed * dt + 0.5 * acceleration * dt ** 2)
    if next_speed <= 0.0:
        next_speed = 0.0
        acceleration = 0.0
    return next_position, next_speed, acceleration


def cycle_reference(traffic_map, sim_t, ego_s_init, initial_gap):
    speed, distance, acceleration = traffic_map.find_speed_profile_information(sim_t=sim_t)
    return {
        "speed": float(speed),
        "world_s": float(distance + ego_s_init + initial_gap),
        "acceleration": float(acceleration),
    }


def build_front_preview(
    traffic_map,
    sim_t,
    preview_dt,
    use_preview,
    front_s,
    front_v,
    front_a,
    ego_s_init,
    initial_gap,
    speed_limit,
    acc_min=-6.0,
    acc_max=4.0,
    horizon=40,
):
    positions = [0.0] * horizon
    speeds = [0.0] * horizon
    accelerations = [0.0] * horizon
    positions[0] = round(float(front_s), 3)
    speeds[0] = round(float(front_v), 3)
    accelerations[0] = round(float(front_a), 3)

    for index in range(1, min(20, horizon)):
        if use_preview:
            speed, distance, acceleration = traffic_map.find_front_vehicle_predicted_state(
                dt=index * preview_dt,
                sim_t=sim_t,
            )
            positions[index] = round(float(distance + ego_s_init + initial_gap), 3)
            speeds[index] = round(float(np.clip(speed, 0.0, speed_limit)), 3)
            accelerations[index] = round(float(np.clip(acceleration, acc_min, acc_max)), 3)
        else:
            positions[index], speeds[index], accelerations[index] = clamp_vehicle_state(
                positions[index - 1],
                speeds[index - 1],
                accelerations[index - 1],
                preview_dt,
                speed_limit,
                acc_min,
                acc_max,
            )
            positions[index] = round(positions[index], 3)
            speeds[index] = round(speeds[index], 3)
            accelerations[index] = round(accelerations[index], 3)
    return positions, speeds, accelerations


def update_idm_followers(
    traffic_manager,
    idm_control,
    dt,
    first_follower_id,
    num_vehicles,
    speed_limit,
    acc_min,
    acc_max,
    cbf_enable=True,
    cbf_alpha=8.0,
    cbf_s0=6.0,
    cbf_time_headway=0.5,
    emergency_decel_margin=2.0,
):
    """Advance followers using bounded IDM with an optional CBF safety filter."""
    for vehicle_id in range(first_follower_id, num_vehicles):
        leader_id = vehicle_id - 1
        commanded_acc = idm_control.safe_IDM_acceleration(
            front_v=traffic_manager.traffic_v[leader_id],
            ego_v=traffic_manager.traffic_v[vehicle_id],
            front_s=traffic_manager.traffic_s[leader_id],
            ego_s=traffic_manager.traffic_s[vehicle_id],
            fallback_acc=acc_min,
        )
        commanded_acc, _, _, _ = idm_control.CBF_acceleration_filter(
            commanded_acc=commanded_acc,
            front_v=traffic_manager.traffic_v[leader_id],
            ego_v=traffic_manager.traffic_v[vehicle_id],
            front_s=traffic_manager.traffic_s[leader_id],
            ego_s=traffic_manager.traffic_s[vehicle_id],
            acc_min=acc_min,
            acc_max=acc_max,
            cbf_enable=cbf_enable,
            cbf_alpha=cbf_alpha,
            cbf_s0=cbf_s0,
            cbf_T=cbf_time_headway,
            emergency_decel_margin=emergency_decel_margin,
        )
        if traffic_manager.traffic_v[vehicle_id] <= 0.0 and commanded_acc < 0.0:
            commanded_acc = 0.0
        position, speed, acceleration = clamp_vehicle_state(
            traffic_manager.traffic_s[vehicle_id],
            traffic_manager.traffic_v[vehicle_id],
            commanded_acc,
            dt,
            speed_limit,
            acc_min,
            acc_max,
        )
        traffic_manager.traffic_s[vehicle_id] = position
        traffic_manager.traffic_v[vehicle_id] = speed
        traffic_manager.traffic_alon[vehicle_id] = acceleration
