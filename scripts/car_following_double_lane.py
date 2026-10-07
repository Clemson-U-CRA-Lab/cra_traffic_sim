#! /usr/bin/env python3

import os
import time

import numpy as np
import rospy
from hololens_ros_communication.msg import ref_traj_correction
from std_msgs.msg import Int8

from double_lane_scenario import DoubleLaneScenario, ScenarioSession, LaneMapGeometry

from sim_env_manager import CMI_traffic_sim, hololens_message_manager, road_reader
from traffic_runtime import (
    build_front_preview,
    cycle_reference,
    get_bool_param,
    update_idm_followers,
    compute_vehicle_following_acceleration,
    update_vehicle_against_stationary,
)
from sim_env_manager import lateral_vehicle_controller
from utils import IDM, host_vehicle_coordinate_transformation


RAD_TO_DEGREE = 180.0 / np.pi


def road_reference_correction_msg_prep(ego_pitch):
    message = ref_traj_correction()
    message.road_ref_x = 0.0
    message.road_ref_y = 0.0
    message.road_ref_z = 0.0
    message.road_ref_pitch = ego_pitch
    message.road_ref_yaw = 0.0
    return message


def main_double_lane_following():
    package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    rospy.init_node("CRA_Digital_Twin_Traffic")
    scenario_file = os.path.expanduser(rospy.get_param(
        "~scenario_file", os.path.join(package_dir, "config", "double_lane_scenario.json")
    ))
    if not os.path.isabs(scenario_file):
        scenario_file = os.path.join(package_dir, scenario_file)
    try:
        scenario = DoubleLaneScenario.load(scenario_file)
    except ValueError as error:
        rospy.logfatal("%s", error)
        return
    num_vehicles = len(scenario.vehicles)
    stationary_vehicle_id = scenario.stationary_id
    moving_vehicle_count = stationary_vehicle_id
    rospy.loginfo("Initial lane ordering (rear to front): %s; predecessors: %s; ego leader: %s",
                  scenario.lane_order, scenario.predecessors, scenario.ego_leader)
    track_style = rospy.get_param("/track_style", "Rally")
    closed_track = track_style == "GrandPrix"
    map_files = [os.path.join(package_dir, "maps", rospy.get_param(name))
                 for name in ("/map_0", "/map_1")]
    speed_profile_file = os.path.join(package_dir, "speed_profile", rospy.get_param("/spd_map"))
    run_sim = get_bool_param(rospy, "/run_sim", False)
    preview_dt = float(rospy.get_param("/pv_states_dt", 0.5))
    use_preview = get_bool_param(rospy, "/use_preview", False)
    run_direction = int(rospy.get_param("/runDirection", 0))
    use_acceleration_pitch = get_bool_param(rospy, "/use_acceleration_pitch", run_sim)

    rate = rospy.Rate(100)
    traffic_manager = CMI_traffic_sim(12, num_vehicles, sil_simulation=run_sim)
    # Load metadata now; positioned messages wait for a valid ego pose.
    scenario.initialize(traffic_manager, 0.0)
    hololens_manager = hololens_message_manager(
        num_vehicles=num_vehicles,
        max_num_vehicles=200,
        max_num_traffic_lights=12,
        num_traffic_lights=0,
    )
    lane_maps = [road_reader(filename, speed_profile_file, closed_track=closed_track)
                 for filename in map_files]
    geometry = LaneMapGeometry(lane_maps)
    traffic_map = lane_maps[0]  # Ego reference, driving cycle, and target lane.
    for lane_map in lane_maps:
        lane_map.read_map_data()
    idm = IDM(a=4, b=6, s0=6, v0=35, T=0.8)
    traffic_map.read_speed_profile()

    direction_pub = rospy.Publisher("/runDirection", Int8, queue_size=2)
    heartbeat_pub = rospy.Publisher("/low_level_heartbeat", Int8, queue_size=2)
    road_ref_pub = rospy.Publisher("/ref_traj_correction", ref_traj_correction, queue_size=2)

    message_id = 0
    sim_t = 0.0
    ego_s_init = 0.0
    initial_gap = scenario.front_gap
    leader_offset = scenario.leader_offset
    speed_limit = float(rospy.get_param("/front_vehicle_speed_limit", 35.0)) * 0.44704
    side_speed_limit = float(rospy.get_param("/side_lane_speed_limit", speed_limit))
    front_acc_min = float(rospy.get_param("/front_vehicle_acceleration_lower_limit", -6.0))
    front_acc_max = float(rospy.get_param("/front_vehicle_acceleration_upper_limit", 4.0))
    side_acc_min = float(rospy.get_param("/side_lane_acceleration_lower_limit", -6.0))
    side_acc_max = float(rospy.get_param("/side_lane_acceleration_upper_limit", 3.0))
    cbf_enable = get_bool_param(rospy, "/side_lane_cbf_enable", True)
    cbf_alpha = float(rospy.get_param("/side_lane_cbf_alpha", 1.0))
    cbf_s0 = float(rospy.get_param("/side_lane_cbf_s0", 5.0))
    cbf_time_headway = float(rospy.get_param("/side_lane_cbf_T", 0.8))
    emergency_decel_margin = float(rospy.get_param("/side_lane_cbf_emergency_decel_margin", 1.0))
    stationary_safety_buffer = max(
        float(rospy.get_param("/side_lane_stationary_safety_buffer", 2.0)),
        0.0,
    )
    lane_change_vehicle_id = int(rospy.get_param("/side_lane_overtake_vehicle_id", 2))
    lane_change_duration = max(
        float(rospy.get_param("/side_lane_change_duration", 4.0)),
        0.1,
    )
    max_steering_rate = float(rospy.get_param("/max_steering_rate", 2.0))
    max_jerk = float(rospy.get_param("/max_jerk", 5.0))
    min_lookahead_distance = float(rospy.get_param("/min_lookahead_distance", 15.0))
    if not all(np.isfinite(value) and value >= 0.0 for value in (max_steering_rate, max_jerk)):
        raise ValueError("max_steering_rate and max_jerk must be finite and nonnegative")
    if not np.isfinite(min_lookahead_distance) or min_lookahead_distance <= 0.0:
        raise ValueError("min_lookahead_distance must be finite and positive")
    lane_change_progress = 0.0
    lane_change_active = False
    lane_change_controller = None
    obstacle_following = False
    ego_pitch = 0.0
    initial_speed, initial_profile_distance, _ = traffic_map.find_speed_profile_information(sim_t=0.0)
    session = ScenarioSession(scenario, float(np.clip(initial_speed, 0.0, speed_limit)))
    front_s = [0.0] * 40
    front_v = [0.0] * 40
    front_a = [0.0] * 40

    while not rospy.is_shutdown():
        try:
            # Snapshot the joystick request so a callback cannot split an update.
            requested_start = traffic_manager.sim_start
            pose_ready = traffic_manager.ego_pose_received
            ego_s = 0.0
            if pose_ready:
                ego_s, _, _ = traffic_map.find_ego_vehicle_distance_reference(traffic_manager.ego_pose_ref)
                pose_ready = bool(np.isfinite(ego_s))
                if pose_ready and not session.started:
                    pose_ready = geometry.align_to_ego(traffic_manager.ego_pose_ref, ego_s)
            dt = session.update(traffic_manager, ego_s, pose_ready, requested_start, time.monotonic())
            if not session.ready:
                rospy.logwarn_throttle(5.0, "Waiting for a valid ego pose on both maps before publishing the traffic scene")
                rate.sleep()
                continue
            sim_t = session.sim_t
            stationary_vehicle_s = traffic_manager.traffic_s[stationary_vehicle_id]
            if session.started:
                # Shared cycle/preview helpers add the raw profile distance.
                ego_s_init = session.anchor_s - initial_profile_distance
            traffic_manager.serial_id = message_id
            hololens_manager.update_ego_state(
                serial_id=message_id,
                ego_x=traffic_manager.ego_x,
                ego_y=traffic_manager.ego_y,
                ego_z=traffic_manager.ego_z,
                ego_yaw=traffic_manager.ego_yaw,
                ego_pitch=traffic_manager.ego_pitch,
                ego_v=traffic_manager.ego_v,
                ego_acc=traffic_manager.ego_acc,
                ego_omega=traffic_manager.ego_omega,
            )

            ego_reference = geometry.pose(ego_s, lane_id=0)
            lane_change_pose = None
            direction_pub.publish(Int8(data=run_direction))
            heartbeat_pub.publish(Int8(data=1))

            message_id += 1
            if session.running:
                cycle = cycle_reference(traffic_map, sim_t, ego_s_init, initial_gap)
                cycle["speed"] = float(np.clip(cycle["speed"], 0.0, speed_limit))
                cycle["acceleration"] = float(np.clip(cycle["acceleration"], front_acc_min, front_acc_max))
                traffic_manager.traffic_update_from_spd_profile(
                    cycle["world_s"], cycle["speed"], cycle["acceleration"], 0
                )
                front_s, front_v, front_a = build_front_preview(
                    traffic_map,
                    sim_t,
                    preview_dt,
                    use_preview,
                    traffic_manager.traffic_s[0],
                    traffic_manager.traffic_v[0],
                    traffic_manager.traffic_alon[0],
                    ego_s_init,
                    initial_gap,
                    speed_limit,
                    front_acc_min,
                    front_acc_max,
                )

                if moving_vehicle_count > 1:
                    obstacle_following = update_vehicle_against_stationary(
                        traffic_manager=traffic_manager,
                        idm_control=idm,
                        vehicle_id=1,
                        stationary_s=stationary_vehicle_s,
                        dt=dt,
                        speed_limit=side_speed_limit,
                        acc_min=side_acc_min,
                        acc_max=side_acc_max,
                        cbf_s0=cbf_s0,
                        cbf_time_headway=cbf_time_headway,
                        cbf_alpha=cbf_alpha,
                        emergency_decel_margin=emergency_decel_margin,
                        detection_buffer=stationary_safety_buffer,
                        following_active=obstacle_following,
                    )
                    if not obstacle_following:
                        traffic_manager.traffic_s[1] = traffic_manager.traffic_s[0] + leader_offset
                        traffic_manager.traffic_v[1] = traffic_manager.traffic_v[0]
                        traffic_manager.traffic_alon[1] = traffic_manager.traffic_alon[0]
                    if (
                        moving_vehicle_count > lane_change_vehicle_id
                        and lane_change_vehicle_id == 2
                        and obstacle_following
                    ):
                        if not lane_change_active:
                            side_pose = geometry.pose(
                                traffic_manager.traffic_s[lane_change_vehicle_id],
                                lane_id=1,
                            )
                            lane_change_controller = lateral_vehicle_controller(
                                x_init=side_pose[0],
                                y_init=side_pose[1],
                                z_init=side_pose[2],
                                yaw_init=side_pose[3],
                                pitch_init=side_pose[4],
                                car_length=float(rospy.get_param("/car_length", 3.5)),
                                max_steering_rate=max_steering_rate,
                                max_jerk=max_jerk,
                            )
                            lane_change_controller.v = traffic_manager.traffic_v[lane_change_vehicle_id]
                            lane_change_controller.acc = traffic_manager.traffic_alon[lane_change_vehicle_id]
                        lane_change_active = True

                    if moving_vehicle_count > 2 and lane_change_active:
                        lane_change_progress = min(
                            1.0,
                            lane_change_progress + dt / lane_change_duration,
                        )
                        commanded_acc = compute_vehicle_following_acceleration(
                            traffic_manager,
                            idm,
                            vehicle_id=lane_change_vehicle_id,
                            leader_id=0,
                            acc_min=side_acc_min,
                            acc_max=side_acc_max,
                            cbf_enable=cbf_enable,
                            cbf_alpha=cbf_alpha,
                            cbf_s0=cbf_s0,
                            cbf_time_headway=cbf_time_headway,
                            emergency_decel_margin=emergency_decel_margin,
                        )
                        lane_change_goal_s = traffic_manager.traffic_s[lane_change_vehicle_id] + max(
                            min_lookahead_distance,
                            traffic_manager.traffic_v[lane_change_vehicle_id] * 0.6,
                        )
                        lane_change_goal = geometry.pose(
                            lane_change_goal_s,
                            lane_id=0,
                        )
                        lane_change_controller.pure_pursuit_controller(lane_change_goal)
                        # Advance horizontal motion first; sample road height at the new position below.
                        lane_change_controller.update_vehicle_state(
                            acc=commanded_acc,
                            z=lane_change_controller.z,
                            pitch=lane_change_controller.pitch,
                            dt=dt,
                        )
                        lane_change_controller.v = float(
                            np.clip(lane_change_controller.v, 0.0, side_speed_limit)
                        )
                        lane_change_pose = lane_change_controller.get_traffic_pose()
                        lane_change_s, _, _ = traffic_map.find_ego_vehicle_distance_reference(
                            np.array([[lane_change_pose[0]], [lane_change_pose[1]], [lane_change_pose[2]]])
                        )
                        # Keep following the original side-lane elevation profile, even while
                        # steering toward lane 0. The lookahead point supplies steering only.
                        original_path_pose = geometry.pose(lane_change_s, lane_id=1)
                        lane_change_controller.z = original_path_pose[2]
                        lane_change_controller.pitch = original_path_pose[4]
                        traffic_manager.traffic_s[lane_change_vehicle_id] = lane_change_s
                        traffic_manager.traffic_v[lane_change_vehicle_id] = lane_change_controller.v
                        traffic_manager.traffic_alon[lane_change_vehicle_id] = lane_change_controller.acc
                        traffic_manager.traffic_l[lane_change_vehicle_id] = 1.0 - lane_change_progress
                        update_idm_followers(
                            traffic_manager,
                            idm,
                            dt,
                            first_follower_id=3,
                            num_vehicles=moving_vehicle_count,
                            speed_limit=side_speed_limit,
                            acc_min=side_acc_min,
                            acc_max=side_acc_max,
                            cbf_enable=cbf_enable,
                            cbf_alpha=cbf_alpha,
                            cbf_s0=cbf_s0,
                            cbf_time_headway=cbf_time_headway,
                            emergency_decel_margin=emergency_decel_margin,
                            leader_ids={3: 1},
                        )
                    else:
                        update_idm_followers(
                            traffic_manager,
                            idm,
                            dt,
                            first_follower_id=2,
                            num_vehicles=moving_vehicle_count,
                            speed_limit=side_speed_limit,
                            acc_min=side_acc_min,
                            acc_max=side_acc_max,
                            cbf_enable=cbf_enable,
                            cbf_alpha=cbf_alpha,
                            cbf_s0=cbf_s0,
                            cbf_time_headway=cbf_time_headway,
                            emergency_decel_margin=emergency_decel_margin,
                        )

                if use_acceleration_pitch:
                    ego_pitch = traffic_manager.ego_acceleration_pitch_update(
                        pitch_max=1.6 / RAD_TO_DEGREE,
                        pitch_min=-1.6 / RAD_TO_DEGREE,
                        acc_max=8.0,
                        acc_min=-9.0,
                    )
                else:
                    ego_pitch = 0.0

            ego_vehicle = [
                traffic_manager.ego_x,
                traffic_manager.ego_y,
                ego_reference[2],
                traffic_manager.ego_yaw,
                ego_reference[4],
            ]
            _, yaw_s, longitudinal_speed, lateral_speed = traffic_map.find_ego_frenet_pose(
                traffic_manager.ego_pose_ref,
                ego_vehicle[3],
                traffic_manager.ego_v_north,
                traffic_manager.ego_v_east,
            )
            traffic_manager.ego_vehicle_frenet_update(
                s=ego_s,
                l=0.0,
                sv=longitudinal_speed,
                lv=lateral_speed,
                yaw_s=yaw_s,
            )
            for vehicle_id in range(num_vehicles):
                lane_id = traffic_manager.traffic_l[vehicle_id]
                if vehicle_id == lane_change_vehicle_id and lane_change_controller is not None:
                    vehicle_pose = lane_change_controller.get_traffic_pose()
                else:
                    vehicle_pose = geometry.pose(
                        traffic_manager.traffic_s[vehicle_id], lane_id=lane_id
                    )
                local_pose = host_vehicle_coordinate_transformation(vehicle_pose, ego_vehicle)
                if vehicle_id == stationary_vehicle_id:
                    traffic_manager.traffic_brake_status[vehicle_id] = True
                else:
                    traffic_manager.traffic_brake_status_update(vehicle_id)
                hololens_manager.update_virtual_vehicle_state(
                    vehicle_id=vehicle_id,
                    vehicle_type=traffic_manager.traffic_type[vehicle_id],
                    x=local_pose[0],
                    y=-local_pose[1],
                    z=local_pose[2],
                    yaw=-local_pose[3],
                    pitch=-local_pose[4],
                    acc=traffic_manager.traffic_alon[vehicle_id],
                    vx=traffic_manager.traffic_v[vehicle_id],
                    vy=0.0,
                    brake_status=traffic_manager.traffic_brake_status[vehicle_id],
                )

            if not session.started:
                front_s, front_v, front_a = build_front_preview(
                    traffic_map, 0.0, preview_dt, False,
                    traffic_manager.traffic_s[0], 0.0, 0.0,
                    0.0, 0.0, speed_limit,
                )

            hololens_manager.construct_hololens_info_msg()
            traffic_manager.construct_traffic_sim_info_msg(sim_t=sim_t)
            traffic_manager.construct_vehicle_state_sequence_msg(
                id=message_id,
                t=sim_t,
                s=front_s,
                v=front_v,
                a=front_a,
                sim_start=session.running,
            )
            hololens_manager.publish_virtual_sim_info()
            traffic_manager.publish_traffic_sim_info()
            traffic_manager.publish_vehicle_traj()
            road_ref_pub.publish(road_reference_correction_msg_prep(ego_pitch))
        except (IndexError, RuntimeError) as error:
            rospy.logwarn_throttle(2.0, "Traffic simulation update failed: %s", error)
        rate.sleep()


if __name__ == "__main__":
    main_double_lane_following()
