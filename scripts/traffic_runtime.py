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
    leader_ids=None,
):
    """Advance followers using bounded IDM with an optional CBF safety filter."""
    for vehicle_id in range(first_follower_id, num_vehicles):
        leader_id = (leader_ids or {}).get(vehicle_id, vehicle_id - 1)
        update_vehicle_following(
            traffic_manager=traffic_manager,
            idm_control=idm_control,
            vehicle_id=vehicle_id,
            leader_id=leader_id,
            dt=dt,
            speed_limit=speed_limit,
            acc_min=acc_min,
            acc_max=acc_max,
            cbf_enable=cbf_enable,
            cbf_alpha=cbf_alpha,
            cbf_s0=cbf_s0,
            cbf_time_headway=cbf_time_headway,
            emergency_decel_margin=emergency_decel_margin,
        )


def update_vehicle_following(
    traffic_manager,
    idm_control,
    vehicle_id,
    leader_id,
    dt,
    speed_limit,
    acc_min,
    acc_max,
    cbf_enable=True,
    cbf_alpha=8.0,
    cbf_s0=6.0,
    cbf_time_headway=0.5,
    emergency_decel_margin=2.0,
):
    """Advance one vehicle against an explicitly selected longitudinal leader."""
    commanded_acc = compute_vehicle_following_acceleration(
        traffic_manager=traffic_manager,
        idm_control=idm_control,
        vehicle_id=vehicle_id,
        leader_id=leader_id,
        acc_min=acc_min,
        acc_max=acc_max,
        cbf_enable=cbf_enable,
        cbf_alpha=cbf_alpha,
        cbf_s0=cbf_s0,
        cbf_time_headway=cbf_time_headway,
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


def compute_vehicle_following_acceleration(
    traffic_manager,
    idm_control,
    vehicle_id,
    leader_id,
    acc_min,
    acc_max,
    cbf_enable=True,
    cbf_alpha=8.0,
    cbf_s0=6.0,
    cbf_time_headway=0.5,
    emergency_decel_margin=2.0,
):
    """Return bounded IDM/CBF acceleration without integrating the vehicle."""
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
    return float(np.clip(commanded_acc, acc_min, acc_max))


def update_vehicle_against_stationary(
    traffic_manager,
    idm_control,
    vehicle_id,
    stationary_s,
    dt,
    speed_limit,
    acc_min,
    acc_max,
    cbf_s0=6.0,
    cbf_time_headway=0.5,
    cbf_alpha=8.0,
    emergency_decel_margin=2.0,
    detection_buffer=2.0,
    following_active=False,
):
    """Detect a stationary leader, or keep following it after initial detection.

    Pass the previous return value as following_active to latch obstacle following
    until the scenario is reset, even as braking shrinks the detection envelope.
    """
    ego_s = float(traffic_manager.traffic_s[vehicle_id])
    ego_v = max(float(traffic_manager.traffic_v[vehicle_id]), 0.0)
    gap = float(stationary_s) - ego_s
    max_deceleration = max(abs(float(acc_min)), 1e-3)
    stopping_distance = ego_v ** 2 / (2.0 * max_deceleration)
    time_headway_distance = max(float(cbf_s0), 0.0) + max(float(cbf_time_headway), 0.0) * ego_v
    detection_distance = max(stopping_distance, time_headway_distance) + max(float(detection_buffer), 0.0)

    if not following_active and gap > detection_distance:
        return False

    commanded_acc = idm_control.safe_IDM_acceleration(
        front_v=0.0,
        ego_v=ego_v,
        front_s=stationary_s,
        ego_s=ego_s,
        fallback_acc=acc_min,
    )
    commanded_acc, _, _, _ = idm_control.CBF_acceleration_filter(
        commanded_acc=commanded_acc,
        front_v=0.0,
        ego_v=ego_v,
        front_s=stationary_s,
        ego_s=ego_s,
        acc_min=acc_min,
        acc_max=acc_max,
        cbf_enable=True,
        cbf_alpha=cbf_alpha,
        cbf_s0=cbf_s0,
        cbf_T=cbf_time_headway,
        emergency_decel_margin=emergency_decel_margin,
    )
    commanded_acc = float(np.clip(commanded_acc, acc_min, acc_max))
    dt = max(float(dt), 0.0)
    if ego_v == 0.0 and commanded_acc < 0.0:
        commanded_acc = 0.0
    next_s, next_v, next_a = clamp_vehicle_state(
        ego_s,
        ego_v,
        commanded_acc,
        dt,
        speed_limit,
        acc_min,
        acc_max,
    )

    # Integrate only up to the stopping instant, not backward for the rest of dt.
    if commanded_acc < 0.0 and ego_v + commanded_acc * dt <= 0.0:
        stop_dt = ego_v / -commanded_acc
        next_s = ego_s + ego_v * stop_dt + 0.5 * commanded_acc * stop_dt ** 2
        next_v = 0.0
        next_a = 0.0

    clearance = max(float(cbf_s0), 0.0)
    if next_s >= float(stationary_s) - clearance:
        next_s = min(next_s, float(stationary_s) - clearance)
        next_v = 0.0
        next_a = acc_min

    traffic_manager.traffic_s[vehicle_id] = next_s
    traffic_manager.traffic_v[vehicle_id] = next_v
    traffic_manager.traffic_alon[vehicle_id] = next_a
    return True
