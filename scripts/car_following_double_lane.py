#! /usr/bin/env python3

import os
import time

import numpy as np
import rospy
from hololens_ros_communication.msg import ref_traj_correction
from std_msgs.msg import Int8

from sim_env_manager import CMI_traffic_sim, hololens_message_manager, road_reader
from traffic_runtime import (
    build_front_preview,
    cycle_reference,
    get_bool_param,
    update_idm_followers,
)
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
    num_vehicles = int(rospy.get_param("/num_vehicles"))
    track_style = rospy.get_param("/track_style", "Rally")
    closed_track = track_style == "GrandPrix"
    map_file = os.path.join(package_dir, "maps", rospy.get_param("/map"))
    speed_profile_file = os.path.join(package_dir, "speed_profile", rospy.get_param("/spd_map"))
    run_sim = get_bool_param(rospy, "/run_sim", False)
    preview_dt = float(rospy.get_param("/pv_states_dt", 0.5))
    use_preview = get_bool_param(rospy, "/use_preview", False)
    run_direction = int(rospy.get_param("/runDirection", 0))
    use_acceleration_pitch = get_bool_param(rospy, "/use_acceleration_pitch", run_sim)

    rospy.init_node("CRA_Digital_Twin_Traffic")
    rate = rospy.Rate(100)
    traffic_manager = CMI_traffic_sim(12, num_vehicles, sil_simulation=run_sim)
    hololens_manager = hololens_message_manager(
        num_vehicles=num_vehicles,
        max_num_vehicles=200,
        max_num_traffic_lights=12,
        num_traffic_lights=0,
    )
    traffic_map = road_reader(map_file, speed_profile_file, closed_track=closed_track)
    idm = IDM(a=4, b=6, s0=6, v0=35, T=0.8)
    traffic_map.read_map_data()
    traffic_map.read_speed_profile()

    direction_pub = rospy.Publisher("/runDirection", Int8, queue_size=2)
    heartbeat_pub = rospy.Publisher("/low_level_heartbeat", Int8, queue_size=2)
    road_ref_pub = rospy.Publisher("/ref_traj_correction", ref_traj_correction, queue_size=2)

    message_id = 0
    sim_t = 0.0
    previous_time = time.time()
    ego_s_init = 0.0
    initial_gap = 8.0
    leader_offset = max(float(rospy.get_param("/side_lane_leader_distance_offset", 12.0)), 0.0)
    speed_limit = float(rospy.get_param("/front_vehicle_speed_limit", 35.0)) * 0.44704
    side_speed_limit = float(rospy.get_param("/side_lane_speed_limit", speed_limit))
    front_acc_min = float(rospy.get_param("/front_vehicle_acceleration_lower_limit", -6.0))
    front_acc_max = float(rospy.get_param("/front_vehicle_acceleration_upper_limit", 4.0))
    side_acc_min = float(rospy.get_param("/side_lane_acceleration_lower_limit", -6.0))
    side_acc_max = float(rospy.get_param("/side_lane_acceleration_upper_limit", 3.0))
    cbf_enable = get_bool_param(rospy, "/side_lane_cbf_enable", True)
    cbf_alpha = float(rospy.get_param("/side_lane_cbf_alpha", 8.0))
    cbf_s0 = float(rospy.get_param("/side_lane_cbf_s0", 6.0))
    cbf_time_headway = float(rospy.get_param("/side_lane_cbf_T", 0.5))
    emergency_decel_margin = float(rospy.get_param("/side_lane_cbf_emergency_decel_margin", 2.0))
    ego_pitch = 0.0

    while not rospy.is_shutdown():
        try:
            dt = max(time.time() - previous_time, 0.0)
            previous_time = time.time()
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

            ego_s, _, _ = traffic_map.find_ego_vehicle_distance_reference(traffic_manager.ego_pose_ref)
            ego_reference = traffic_map.find_traffic_vehicle_poses(ego_s, lane_id=0)
            front_s = [0.0] * 40
            front_v = [0.0] * 40
            front_a = [0.0] * 40
            direction_pub.publish(Int8(data=run_direction))
            heartbeat_pub.publish(Int8(data=1))

            if sim_t < 0.5 and traffic_manager.sim_start:
                sim_t += dt
                initial_speed, _, _ = traffic_map.find_speed_profile_information(sim_t=0.0)
                traffic_manager.traffic_initialization(
                    ego_s,
                    initial_gap,
                    0,
                    0,
                    0,
                    initial_speed=initial_speed,
                    initial_acceleration=0.0,
                )
                for vehicle_id in range(1, num_vehicles):
                    side_lane_s = (
                        traffic_manager.traffic_s[0]
                        + leader_offset
                        - initial_gap * (vehicle_id - 1)
                    )
                    traffic_manager.traffic_initialization(
                        side_lane_s,
                        0.0,
                        1,
                        vehicle_id,
                        0,
                        initial_speed=initial_speed,
                        initial_acceleration=0.0,
                    )
                ego_s_init = ego_s
                continue

            message_id += 1
            if traffic_manager.sim_start:
                sim_t += dt
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

                if num_vehicles > 1:
                    traffic_manager.traffic_s[1] = traffic_manager.traffic_s[0] + leader_offset
                    traffic_manager.traffic_v[1] = traffic_manager.traffic_v[0]
                    traffic_manager.traffic_alon[1] = traffic_manager.traffic_alon[0]
                    update_idm_followers(
                        traffic_manager,
                        idm,
                        dt,
                        first_follower_id=2,
                        num_vehicles=num_vehicles,
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
                    lane_id = 0 if vehicle_id == 0 else 1
                    vehicle_pose = traffic_map.find_traffic_vehicle_poses(
                        traffic_manager.traffic_s[vehicle_id], lane_id=lane_id
                    )
                    local_pose = host_vehicle_coordinate_transformation(vehicle_pose, ego_vehicle)
                    traffic_manager.traffic_brake_status_update(vehicle_id)
                    hololens_manager.update_virtual_vehicle_state(
                        vehicle_id=vehicle_id,
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
            else:
                for vehicle_id in range(num_vehicles):
                    lane_id = 0 if vehicle_id == 0 else 1
                    vehicle_pose = traffic_map.find_traffic_vehicle_poses(
                        traffic_manager.traffic_s[vehicle_id] - ego_s, lane_id=lane_id
                    )
                    ego_vehicle = [
                        traffic_manager.ego_x,
                        traffic_manager.ego_y,
                        ego_reference[2],
                        traffic_manager.ego_yaw,
                        ego_reference[4],
                    ]
                    local_pose = host_vehicle_coordinate_transformation(vehicle_pose, ego_vehicle)
                    hololens_manager.update_virtual_vehicle_state(
                        vehicle_id=vehicle_id,
                        x=local_pose[0],
                        y=-local_pose[1],
                        z=local_pose[2],
                        yaw=-local_pose[3],
                        pitch=-local_pose[4],
                        acc=traffic_manager.traffic_alon[vehicle_id],
                        vx=traffic_manager.traffic_v[vehicle_id],
                        vy=0.0,
                        brake_status=traffic_manager.traffic_alon[vehicle_id] <= 0.0,
                    )

            hololens_manager.construct_hololens_info_msg()
            traffic_manager.construct_traffic_sim_info_msg(sim_t=sim_t)
            traffic_manager.construct_vehicle_state_sequence_msg(
                id=message_id,
                t=sim_t,
                s=front_s,
                v=front_v,
                a=front_a,
                sim_start=traffic_manager.sim_start,
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
