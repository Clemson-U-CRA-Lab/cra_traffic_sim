#! /usr/bin/env python3

import os
import time

import numpy as np
import rospy
from hololens_ros_communication.msg import ref_traj_correction
from std_msgs.msg import Int8

from sim_env_manager import *
from utils import *

# Define constants
RAD_TO_DEGREE = 180.0 / np.pi
DRIVING_CYCLE_MODE = "driving_cycle"
BEHAVIOR_GENERATION_MODE = "behavior_generation"
RETURN_TO_CYCLE_MODE = "return_to_cycle"
STOP_AT_DISTANCE_MODE = "stop_at_distance"
HOLD_STOPPED_MODE = "hold_stopped"


def road_reference_correction_msg_prep(ego_pitch):
    road_ref_correction_msg = ref_traj_correction()
    road_ref_correction_msg.road_ref_x = 0.0
    road_ref_correction_msg.road_ref_y = 0.0
    road_ref_correction_msg.road_ref_z = 0.0
    road_ref_correction_msg.road_ref_pitch = ego_pitch
    road_ref_correction_msg.road_ref_yaw = 0.0
    return road_ref_correction_msg


def clamp_vehicle_state(front_s, front_v, front_a, dt, speed_limit, acc_min, acc_max):
    clipped_acc = float(np.clip(front_a, acc_min, acc_max))
    next_v = float(np.clip(front_v + clipped_acc * dt, 0.0, speed_limit))
    next_s = float(front_s + front_v * dt + 0.5 * clipped_acc * dt ** 2)
    if next_v <= 0.0:
        next_v = 0.0
        clipped_acc = 0.0
    return next_s, next_v, clipped_acc


def get_cycle_reference(traffic_map_manager, sim_t, ego_s_init, init_gap):
    speed_t, dist_t, acc_t = traffic_map_manager.find_speed_profile_information(sim_t=sim_t)
    return {
        "speed": float(speed_t),
        "dist": float(dist_t),
        "acc": float(acc_t),
        "world_s": float(dist_t + ego_s_init + init_gap),
    }


def apply_cycle_reference_with_offset(traffic_manager, cycle_ref, distance_offset, vehicle_id=0):
    traffic_manager.traffic_update_from_spd_profile(
        s_t=cycle_ref["world_s"] + distance_offset,
        v_t=cycle_ref["speed"],
        a_t=cycle_ref["acc"],
        vehicle_id=vehicle_id,
    )


def compute_return_to_cycle_acceleration(front_s, front_v, cycle_ref, front_vehicle_speed_limit, dt, acc_min, acc_max):
    dt_safe = max(dt, 1e-3)
    speed_error = cycle_ref["speed"] - front_v
    position_error = cycle_ref["world_s"] - front_s
    target_velocity_from_position = np.clip(position_error / dt_safe, -front_vehicle_speed_limit, front_vehicle_speed_limit)
    velocity_tracking_error = target_velocity_from_position - front_v
    commanded_acc = 0.8 * speed_error + 0.15 * velocity_tracking_error + cycle_ref["acc"]
    return float(np.clip(commanded_acc, acc_min, acc_max))


def should_begin_stop_at_distance(front_s, front_v, target_stop_s, acc_min, stop_buffer):
    if target_stop_s is None:
        return False

    remaining_distance = target_stop_s - front_s
    if remaining_distance <= 0.0:
        return True

    max_deceleration = max(abs(acc_min), 1e-3)
    stopping_distance = front_v ** 2 / (2.0 * max_deceleration)
    return remaining_distance <= stopping_distance + stop_buffer


def compute_stop_at_distance_acceleration(
    front_s,
    front_v,
    target_stop_s,
    acc_min,
    acc_max,
    distance_tolerance,
    speed_tolerance,
):
    remaining_distance = target_stop_s - front_s
    if remaining_distance <= distance_tolerance:
        if front_v > speed_tolerance:
            return float(acc_min)
        return 0.0

    if front_v <= speed_tolerance:
        max_deceleration = max(abs(acc_min), 1e-3)
        approach_speed = min(2.0, np.sqrt(2.0 * max_deceleration * remaining_distance))
        commanded_acc = 0.5 * (approach_speed - front_v)
        return float(np.clip(commanded_acc, 0.0, acc_max))

    required_acc = -(front_v ** 2) / (2.0 * remaining_distance)
    return float(np.clip(required_acc, acc_min, min(0.0, acc_max)))


def clamp_to_stop_target(front_s, front_v, front_a, target_stop_s, distance_tolerance, speed_tolerance):
    if target_stop_s is None:
        return front_s, front_v, front_a, False

    remaining_distance = target_stop_s - front_s
    if front_s >= target_stop_s or (remaining_distance <= distance_tolerance and front_v <= speed_tolerance):
        return float(target_stop_s), 0.0, 0.0, True

    return front_s, front_v, front_a, False


def update_front_vehicle_stop_at_distance(
    front_s,
    front_v,
    dt,
    target_stop_s,
    speed_limit,
    acc_min,
    acc_max,
    distance_tolerance,
    speed_tolerance,
):
    commanded_acc = compute_stop_at_distance_acceleration(
        front_s,
        front_v,
        target_stop_s,
        acc_min,
        acc_max,
        distance_tolerance,
        speed_tolerance,
    )
    next_s, next_v, next_a = clamp_vehicle_state(
        front_s,
        front_v,
        commanded_acc,
        dt,
        speed_limit,
        acc_min,
        acc_max,
    )
    return clamp_to_stop_target(
        next_s,
        next_v,
        next_a,
        target_stop_s,
        distance_tolerance,
        speed_tolerance,
    )


def build_linear_reward_target_window(current_time, horizon_length, time_interval, target_max, ramp_duration):
    if ramp_duration <= 0.0:
        raise ValueError("ramp_duration must be positive.")

    future_time_window = current_time + np.arange(horizon_length) * time_interval
    reward_progress = np.clip(future_time_window / ramp_duration, 0.0, 1.0)
    return target_max * reward_progress


def build_front_preview(
    mode,
    sim_t,
    pv_dt,
    use_preview,
    front_vehicle_speed_limit,
    traffic_map_manager,
    front_s,
    front_v,
    front_a,
    ego_s_init,
    init_gap,
    driving_cycle_distance_offset,
    front_acc_min,
    front_acc_max,
    target_stop_s=None,
    stop_distance_tolerance=0.05,
    stop_speed_tolerance=0.1,
):
    front_s_t = [0.0] * 40
    front_v_t = [0.0] * 40
    front_a_t = [0.0] * 40
    front_s_t[0] = round(front_s, 3)
    front_v_t[0] = round(front_v, 3)
    front_a_t[0] = round(front_a, 3)

    for i in range(1, 20):
        if mode == DRIVING_CYCLE_MODE and use_preview:
            v_t, s_t, a_t = traffic_map_manager.find_front_vehicle_predicted_state(dt=i * pv_dt, sim_t=sim_t)
            predicted_s = s_t + ego_s_init + init_gap + driving_cycle_distance_offset
            if target_stop_s is not None and predicted_s >= target_stop_s:
                front_s_t[i] = round(target_stop_s, 3)
                front_v_t[i] = 0.0
                front_a_t[i] = 0.0
                continue
            front_s_t[i] = round(predicted_s, 3)
            front_v_t[i] = round(v_t, 3)
            front_a_t[i] = round(a_t, 3)
            continue

        prev_s = front_s_t[i - 1]
        prev_v = front_v_t[i - 1]

        if mode == RETURN_TO_CYCLE_MODE:
            cycle_ref = get_cycle_reference(traffic_map_manager, sim_t + i * pv_dt, ego_s_init, init_gap)
            preview_a = compute_return_to_cycle_acceleration(
                prev_s,
                prev_v,
                cycle_ref,
                front_vehicle_speed_limit,
                pv_dt,
                front_acc_min,
                front_acc_max,
            )
        elif mode == STOP_AT_DISTANCE_MODE and target_stop_s is not None:
            preview_a = compute_stop_at_distance_acceleration(
                prev_s,
                prev_v,
                target_stop_s,
                front_acc_min,
                front_acc_max,
                stop_distance_tolerance,
                stop_speed_tolerance,
            )
        else:
            preview_a = front_a_t[i - 1]

        next_s, next_v, next_a = clamp_vehicle_state(
            prev_s,
            prev_v,
            preview_a,
            pv_dt,
            front_vehicle_speed_limit,
            front_acc_min,
            front_acc_max,
        )
        if mode in {STOP_AT_DISTANCE_MODE, HOLD_STOPPED_MODE}:
            next_s, next_v, next_a, _ = clamp_to_stop_target(
                next_s,
                next_v,
                next_a,
                target_stop_s,
                stop_distance_tolerance,
                stop_speed_tolerance,
            )
        front_s_t[i] = round(next_s, 3)
        front_v_t[i] = round(next_v, 3)
        front_a_t[i] = round(next_a, 3)

    return front_s_t, front_v_t, front_a_t


def main_single_lane_following():
    # Path Parameters
    current_dirname = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dirname, os.pardir))

    num_Sv = rospy.get_param("/num_vehicles")
    track_style = rospy.get_param("/track_style")

    if track_style == "GrandPrix":
        closed_loop = True
    if track_style == "Rally":
        closed_loop = False

    map_filename = rospy.get_param("/map")
    spd_filename = rospy.get_param("/spd_map")
    run_sim = bool(rospy.get_param("/run_sim"))
    pv_dt = float(rospy.get_param("/pv_states_dt"))
    use_preview = bool(rospy.get_param("/use_preview"))
    run_direction = rospy.get_param("/runDirection")
    koopman_lift_method = rospy.get_param("/koopman_lift_method", "auto")
    front_vehicle_travel_distance = float(rospy.get_param("/front_vehicle_travel_distance"))
    front_vehicle_stop_distance_tolerance = max(
        float(rospy.get_param("/front_vehicle_stop_distance_tolerance", 0.05)),
        0.0,
    )
    front_vehicle_stop_speed_tolerance = max(
        float(rospy.get_param("/front_vehicle_stop_speed_tolerance", 0.1)),
        0.0,
    )
    front_vehicle_stop_buffer = max(float(rospy.get_param("/front_vehicle_stop_buffer", 2.0)), 0.0)

    map_1_file = os.path.join(parent_dir, "maps", map_filename)
    spd_file = os.path.join(parent_dir, "speed_profile", spd_filename)
    print("PV speed pfofile is: ", spd_file)

    rospy.init_node("CRA_Digital_Twin_Traffic")
    rate = rospy.Rate(100)

    traffic_manager = CMI_traffic_sim(
        max_num_vehicles=12,
        num_vehicles=num_Sv,
        sil_simulation=run_sim,
    )

    virtual_traffic_sim_info_manager = hololens_message_manager(
        num_vehicles=1,
        max_num_vehicles=200,
        max_num_traffic_lights=12,
        num_traffic_lights=0,
    )

    traffic_map_manager = road_reader(
        map_filename=map_1_file,
        speed_profile_filename=spd_file,
        closed_track=closed_loop,
    )

    front_vehicle_motion_generator = preceding_vehicle_spd_profile_generation(
        horizon_length=8,
        time_interval=pv_dt,
    )

    matrix_data_folder = os.path.join(parent_dir, "pv_scenario_generation_workspace", "data_driven_workspace")
    A, B, C = front_vehicle_motion_generator.load_matrices_from_file(
        data_folder_path=matrix_data_folder,
        preferred_lift_method=koopman_lift_method,
    )
    if A is None or B is None or C is None:
        rospy.logerr(f"Failed to load reward-tracking model matrices from {matrix_data_folder}.")
        raise RuntimeError("Reward tracking model data is required for behavior generation.")
    rospy.loginfo(
        "Loaded Koopman model with lift %s (A%s, B%s, C%s).",
        front_vehicle_motion_generator.koopman_lift_method,
        A.shape,
        B.shape,
        C.shape,
    )

    traffic_map_manager.read_map_data()
    traffic_map_manager.read_speed_profile()

    dir_msg_publisher = rospy.Publisher("/runDirection", Int8, queue_size=2)
    lowlevel_heartbeat_publisher = rospy.Publisher("/low_level_heartbeat", Int8, queue_size=2)
    road_ref_pub = rospy.Publisher("/ref_traj_correction", ref_traj_correction, queue_size=2)

    msg_counter = 0
    ego_vehicle_pitch_from_acceleration = 0.0
    prev_t = time.time()
    sim_t = 0.0
    ego_s_init = 0.0
    init_gap = 8.0

    reward_tracking_duration = 10.0
    reward_target_ramp_duration = reward_tracking_duration
    reward_target_max = 20.0
    reward_Q = 1.0
    reward_R = 10.0
    reward_R_du = 100.0
    maintain_motion_duration = 5.0
    behavior_generation_duration = reward_tracking_duration + maintain_motion_duration
    front_vehicle_speed_limit = 22.5
    front_vehicle_acc_max = 4.0
    front_vehicle_acc_min = -4.0
    return_speed_tolerance = 0.3
    driving_cycle_distance_offset = 0.0

    scenario_mode = DRIVING_CYCLE_MODE
    behavior_generation_start_time = None
    front_vehicle_stop_target_s = None
    front_vehicle_stop_target_logged = False

    while not rospy.is_shutdown():
        try:
            Dt = time.time() - prev_t
            prev_t = time.time()

            traffic_manager.serial_id = msg_counter
            virtual_traffic_sim_info_manager.update_ego_state(
                serial_id=traffic_manager.serial_id,
                ego_x=traffic_manager.ego_x,
                ego_y=traffic_manager.ego_y,
                ego_z=traffic_manager.ego_z,
                ego_yaw=traffic_manager.ego_yaw,
                ego_pitch=traffic_manager.ego_pitch,
                ego_v=traffic_manager.ego_v,
                ego_acc=traffic_manager.ego_acc,
                ego_omega=traffic_manager.ego_omega,
            )

            s_ego_frenet, _, _ = traffic_map_manager.find_ego_vehicle_distance_reference(traffic_manager.ego_pose_ref)
            ego_vehicle_ref_poses = traffic_map_manager.find_traffic_vehicle_poses(s_ego_frenet, lane_id=0)

            front_s_t = [0.0] * 40
            front_v_t = [0.0] * 40
            front_a_t = [0.0] * 40

            run_dir_msg = Int8()
            run_dir_msg.data = run_direction
            dir_msg_publisher.publish(run_dir_msg)

            lowlevel_heartbeat_msg = Int8()
            lowlevel_heartbeat_msg.data = 1
            lowlevel_heartbeat_publisher.publish(lowlevel_heartbeat_msg)

            if sim_t < 0.5 and traffic_manager.sim_start:
                sim_t += Dt
                traffic_manager.traffic_initialization(
                    s_ego=s_ego_frenet,
                    ds=init_gap,
                    line_number=0,
                    vehicle_id=0,
                    vehicle_id_in_lane=0,
                )
                ego_s_init = s_ego_frenet
                if front_vehicle_travel_distance > 0.0:
                    front_vehicle_stop_target_s = traffic_manager.traffic_s[0] + front_vehicle_travel_distance
                continue
            else:
                msg_counter += 1
                if traffic_manager.sim_start:
                    sim_t += Dt
                    cycle_ref = get_cycle_reference(traffic_map_manager, sim_t, ego_s_init, init_gap)
                    if front_vehicle_stop_target_s is not None and not front_vehicle_stop_target_logged:
                        rospy.loginfo(
                            "Front vehicle stop target set to %.3f m Frenet s (%.3f m travel distance).",
                            front_vehicle_stop_target_s,
                            front_vehicle_travel_distance,
                        )
                        front_vehicle_stop_target_logged = True

                    if scenario_mode == DRIVING_CYCLE_MODE and traffic_manager.consume_behavior_generation_request():
                        scenario_mode = BEHAVIOR_GENERATION_MODE
                        behavior_generation_start_time = sim_t
                        rospy.loginfo("Front vehicle switched to behavior generation mode.")

                    if front_vehicle_stop_target_s is not None and scenario_mode not in {
                        STOP_AT_DISTANCE_MODE,
                        HOLD_STOPPED_MODE,
                    }:
                        cycle_s_with_offset = cycle_ref["world_s"] + driving_cycle_distance_offset
                        cycle_would_pass_target = (
                            scenario_mode == DRIVING_CYCLE_MODE
                            and cycle_s_with_offset >= front_vehicle_stop_target_s
                        )
                        should_stop_now = should_begin_stop_at_distance(
                            traffic_manager.traffic_s[0],
                            traffic_manager.traffic_v[0],
                            front_vehicle_stop_target_s,
                            front_vehicle_acc_min,
                            front_vehicle_stop_buffer,
                        )
                        if cycle_would_pass_target or should_stop_now:
                            scenario_mode = STOP_AT_DISTANCE_MODE
                            behavior_generation_start_time = None
                            rospy.loginfo("Front vehicle switched to stop-at-distance mode.")

                    if scenario_mode == DRIVING_CYCLE_MODE:
                        apply_cycle_reference_with_offset(
                            traffic_manager=traffic_manager,
                            cycle_ref=cycle_ref,
                            distance_offset=driving_cycle_distance_offset,
                            vehicle_id=0,
                        )
                    elif scenario_mode == BEHAVIOR_GENERATION_MODE:
                        front_vehicle_motion_generator.update_ego_vehicle_state(
                            ego_a_t=traffic_manager.ego_acc,
                            ego_v_t=traffic_manager.ego_v,
                            ego_s_t=traffic_manager.ego_s,
                            pv_a_t=traffic_manager.traffic_alon[0],
                            pv_v_t=traffic_manager.traffic_v[0],
                            pv_s_t=traffic_manager.traffic_s[0],
                        )

                        d_a_max = front_vehicle_acc_max - traffic_manager.ego_acc
                        d_a_min = front_vehicle_acc_min - traffic_manager.ego_acc
                        d_v_max = front_vehicle_speed_limit - traffic_manager.ego_v
                        d_v_min = 0.0 - traffic_manager.ego_v
                        behavior_elapsed = sim_t - behavior_generation_start_time

                        if behavior_elapsed < reward_tracking_duration:
                            reward_target_window = build_linear_reward_target_window(
                                current_time=behavior_elapsed,
                                horizon_length=front_vehicle_motion_generator.h,
                                time_interval=pv_dt,
                                target_max=reward_target_max,
                                ramp_duration=reward_target_ramp_duration,
                            )
                            front_vehicle_motion_generator.perform_nonlinear_optimization_for_reward_tracking(
                                Q=reward_Q,
                                R=reward_R,
                                reward_target=reward_target_window,
                                a_max=d_a_max,
                                a_min=d_a_min,
                                v_max=d_v_max,
                                v_min=d_v_min,
                                R_du=reward_R_du,
                            )
                            front_a_target = traffic_manager.ego_acc + float(front_vehicle_motion_generator.reward_tracking_u_opt[0])
                            front_a = float(np.clip(traffic_manager.traffic_alon[0], front_vehicle_acc_min, front_vehicle_acc_max))
                            commanded_acc = front_a + 0.2 * (front_a_target - front_a)
                            commanded_acc = float(np.clip(commanded_acc, front_vehicle_acc_min, front_vehicle_acc_max))
                            use_behavior_update = True
                        elif behavior_elapsed < behavior_generation_duration:
                            commanded_acc = 0.0
                            use_behavior_update = True
                        else:
                            behavior_generation_start_time = None
                            if (
                                front_vehicle_stop_target_s is not None
                                and should_begin_stop_at_distance(
                                    traffic_manager.traffic_s[0],
                                    traffic_manager.traffic_v[0],
                                    front_vehicle_stop_target_s,
                                    front_vehicle_acc_min,
                                    front_vehicle_stop_buffer,
                                )
                            ):
                                use_behavior_update = False
                                scenario_mode = STOP_AT_DISTANCE_MODE
                                rospy.loginfo("Front vehicle switched to stop-at-distance mode.")
                            else:
                                use_behavior_update = True
                                scenario_mode = RETURN_TO_CYCLE_MODE
                                commanded_acc = compute_return_to_cycle_acceleration(
                                    traffic_manager.traffic_s[0],
                                    traffic_manager.traffic_v[0],
                                    cycle_ref,
                                    front_vehicle_speed_limit,
                                    Dt,
                                    front_vehicle_acc_min,
                                    front_vehicle_acc_max,
                                )
                                rospy.loginfo("Front vehicle switched to return-to-cycle mode.")

                        if use_behavior_update:
                            next_s, next_v, next_a = clamp_vehicle_state(
                                traffic_manager.traffic_s[0],
                                traffic_manager.traffic_v[0],
                                commanded_acc,
                                Dt,
                                front_vehicle_speed_limit,
                                front_vehicle_acc_min,
                                front_vehicle_acc_max,
                            )
                            stopped_at_target = False
                            if front_vehicle_stop_target_s is not None:
                                next_s, next_v, next_a, stopped_at_target = clamp_to_stop_target(
                                    next_s,
                                    next_v,
                                    next_a,
                                    front_vehicle_stop_target_s,
                                    front_vehicle_stop_distance_tolerance,
                                    front_vehicle_stop_speed_tolerance,
                                )
                            traffic_manager.traffic_s[0] = next_s
                            traffic_manager.traffic_v[0] = next_v
                            traffic_manager.traffic_alon[0] = next_a
                            if stopped_at_target:
                                scenario_mode = HOLD_STOPPED_MODE
                                behavior_generation_start_time = None
                                rospy.loginfo("Front vehicle reached stop target and is holding.")
                    elif scenario_mode == STOP_AT_DISTANCE_MODE:
                        next_s, next_v, next_a, stopped_at_target = update_front_vehicle_stop_at_distance(
                            traffic_manager.traffic_s[0],
                            traffic_manager.traffic_v[0],
                            Dt,
                            front_vehicle_stop_target_s,
                            front_vehicle_speed_limit,
                            front_vehicle_acc_min,
                            front_vehicle_acc_max,
                            front_vehicle_stop_distance_tolerance,
                            front_vehicle_stop_speed_tolerance,
                        )
                        traffic_manager.traffic_s[0] = next_s
                        traffic_manager.traffic_v[0] = next_v
                        traffic_manager.traffic_alon[0] = next_a
                        if stopped_at_target:
                            scenario_mode = HOLD_STOPPED_MODE
                            rospy.loginfo("Front vehicle reached stop target and is holding.")
                    elif scenario_mode == HOLD_STOPPED_MODE:
                        traffic_manager.traffic_s[0] = front_vehicle_stop_target_s
                        traffic_manager.traffic_v[0] = 0.0
                        traffic_manager.traffic_alon[0] = 0.0
                    elif scenario_mode == RETURN_TO_CYCLE_MODE:
                        commanded_acc = compute_return_to_cycle_acceleration(
                            traffic_manager.traffic_s[0],
                            traffic_manager.traffic_v[0],
                            cycle_ref,
                            front_vehicle_speed_limit,
                            Dt,
                            front_vehicle_acc_min,
                            front_vehicle_acc_max,
                        )
                        next_s, next_v, next_a = clamp_vehicle_state(
                            traffic_manager.traffic_s[0],
                            traffic_manager.traffic_v[0],
                            commanded_acc,
                            Dt,
                            front_vehicle_speed_limit,
                            front_vehicle_acc_min,
                            front_vehicle_acc_max,
                        )
                        traffic_manager.traffic_s[0] = next_s
                        traffic_manager.traffic_v[0] = next_v
                        traffic_manager.traffic_alon[0] = next_a

                        close_to_cycle_speed = abs(traffic_manager.traffic_v[0] - cycle_ref["speed"]) < return_speed_tolerance
                        if close_to_cycle_speed:
                            driving_cycle_distance_offset = traffic_manager.traffic_s[0] - cycle_ref["world_s"]
                            apply_cycle_reference_with_offset(
                                traffic_manager=traffic_manager,
                                cycle_ref=cycle_ref,
                                distance_offset=driving_cycle_distance_offset,
                                vehicle_id=0,
                            )
                            scenario_mode = DRIVING_CYCLE_MODE
                            behavior_generation_start_time = None
                            rospy.loginfo(
                                "Front vehicle rejoined the driving cycle with %.3f m distance offset.",
                                driving_cycle_distance_offset,
                            )

                    front_s_t, front_v_t, front_a_t = build_front_preview(
                        mode=scenario_mode,
                        sim_t=sim_t,
                        pv_dt=pv_dt,
                        use_preview=use_preview,
                        front_vehicle_speed_limit=front_vehicle_speed_limit,
                        traffic_map_manager=traffic_map_manager,
                        front_s=traffic_manager.traffic_s[0],
                        front_v=traffic_manager.traffic_v[0],
                        front_a=traffic_manager.traffic_alon[0],
                        ego_s_init=ego_s_init,
                        init_gap=init_gap,
                        driving_cycle_distance_offset=driving_cycle_distance_offset,
                        front_acc_min=front_vehicle_acc_min,
                        front_acc_max=front_vehicle_acc_max,
                        target_stop_s=front_vehicle_stop_target_s,
                        stop_distance_tolerance=front_vehicle_stop_distance_tolerance,
                        stop_speed_tolerance=front_vehicle_stop_speed_tolerance,
                    )

                    ego_vehicle_pitch_from_acceleration = traffic_manager.ego_acceleration_pitch_update(
                        pitch_max=2 / RAD_TO_DEGREE,
                        pitch_min=-2 / RAD_TO_DEGREE,
                        acc_max=4.0,
                        acc_min=-6.0,
                    )
                    for i in range(num_Sv):
                        traffic_vehicle_poses = traffic_map_manager.find_traffic_vehicle_poses(traffic_manager.traffic_s[i], lane_id=0)
                        ego_vehicle_poses = [
                            traffic_manager.ego_x,
                            traffic_manager.ego_y,
                            ego_vehicle_ref_poses[2],
                            traffic_manager.ego_yaw,
                            ego_vehicle_ref_poses[4],
                        ]

                        _, yaw_s, v_longitudinal, v_lateral = traffic_map_manager.find_ego_frenet_pose(
                            ego_poses=traffic_manager.ego_pose_ref,
                            ego_yaw=ego_vehicle_poses[3],
                            vy=traffic_manager.ego_v_north,
                            vx=traffic_manager.ego_v_east,
                        )

                        traffic_manager.ego_vehicle_frenet_update(
                            s=s_ego_frenet,
                            l=0,
                            sv=v_longitudinal,
                            lv=v_lateral,
                            yaw_s=yaw_s,
                        )

                        local_traffic_vehicle_poses = host_vehicle_coordinate_transformation(
                            traffic_vehicle_poses,
                            ego_vehicle_poses,
                        )

                        traffic_manager.traffic_brake_status_update(vehicle_id=i)
                        virtual_vehicle_brake = traffic_manager.traffic_brake_status[i]

                        virtual_traffic_sim_info_manager.update_virtual_vehicle_state(
                            vehicle_id=i,
                            x=local_traffic_vehicle_poses[0],
                            y=-local_traffic_vehicle_poses[1],
                            z=local_traffic_vehicle_poses[2],
                            yaw=-local_traffic_vehicle_poses[3],
                            pitch=-local_traffic_vehicle_poses[4],
                            acc=traffic_manager.traffic_alon[i],
                            vx=traffic_manager.traffic_v[i],
                            vy=0.0,
                            brake_status=virtual_vehicle_brake,
                        )
                else:
                    for i in range(num_Sv):
                        traffic_vehicle_poses = traffic_map_manager.find_traffic_vehicle_poses(
                            traffic_manager.traffic_s[i] - s_ego_frenet,
                            lane_id=0,
                        )
                        ego_vehicle_poses = [
                            traffic_manager.ego_x,
                            traffic_manager.ego_y,
                            ego_vehicle_ref_poses[2],
                            traffic_manager.ego_yaw,
                            ego_vehicle_ref_poses[4],
                        ]
                        local_traffic_vehicle_poses = host_vehicle_coordinate_transformation(
                            traffic_vehicle_poses,
                            ego_vehicle_poses,
                        )
                        if traffic_manager.traffic_alon[i] <= 0:
                            virtual_vehicle_brake = True
                        else:
                            virtual_vehicle_brake = False
                        virtual_traffic_sim_info_manager.update_virtual_vehicle_state(
                            vehicle_id=i,
                            x=local_traffic_vehicle_poses[0],
                            y=-local_traffic_vehicle_poses[1],
                            z=local_traffic_vehicle_poses[2],
                            yaw=-local_traffic_vehicle_poses[3],
                            pitch=-local_traffic_vehicle_poses[4],
                            acc=traffic_manager.traffic_alon[i],
                            vx=traffic_manager.traffic_v[i],
                            vy=0.0,
                            brake_status=virtual_vehicle_brake,
                        )

            virtual_traffic_sim_info_manager.construct_hololens_info_msg()
            traffic_manager.construct_traffic_sim_info_msg(sim_t=sim_t)
            traffic_manager.construct_vehicle_state_sequence_msg(
                id=msg_counter,
                t=sim_t,
                s=front_s_t,
                v=front_v_t,
                a=front_a_t,
                sim_start=traffic_manager.sim_start,
            )

            virtual_traffic_sim_info_manager.publish_virtual_sim_info()
            traffic_manager.publish_traffic_sim_info()
            traffic_manager.publish_vehicle_traj()

            road_ref_correction_msg = road_reference_correction_msg_prep(ego_vehicle_pitch_from_acceleration)
            road_ref_pub.publish(road_ref_correction_msg)

        except IndexError:
            print("Index error detected.")
        except RuntimeError:
            print("Runtime error detected.")

        rate.sleep()


if __name__ == "__main__":
    main_single_lane_following()
