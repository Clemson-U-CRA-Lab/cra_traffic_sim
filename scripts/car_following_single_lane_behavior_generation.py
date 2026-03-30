#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from hololens_ros_communication.msg import ref_traj_correction
from std_msgs.msg import Float64MultiArray, Int8


import time
import numpy as np
import os
from sim_env_manager import *
from utils import *

# Define constants
RAD_TO_DEGREE = 52.296

def road_reference_correction_msg_prep(ego_pitch):
    road_ref_correction_msg = ref_traj_correction()
    road_ref_correction_msg.road_ref_x = 0.0
    road_ref_correction_msg.road_ref_y = 0.0
    road_ref_correction_msg.road_ref_z = 0.0
    road_ref_correction_msg.road_ref_pitch = ego_pitch
    road_ref_correction_msg.road_ref_yaw = 0.0
    return road_ref_correction_msg

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

    map_1_file = os.path.join(parent_dir, "maps", map_filename)
    spd_file = os.path.join(parent_dir, "speed_profile", spd_filename)
    print('PV speed pfofile is: ' , spd_file)

    rospy.init_node("CRA_Digital_Twin_Traffic")
    rate = rospy.Rate(100)

    traffic_manager = CMI_traffic_sim(max_num_vehicles=12, 
                                      num_vehicles=num_Sv, 
                                      sil_simulation=run_sim)
    
    virtual_traffic_sim_info_manager = hololens_message_manager(num_vehicles=1, 
                                                                max_num_vehicles=200, 
                                                                max_num_traffic_lights=12, 
                                                                num_traffic_lights=0)
    
    traffic_map_manager = road_reader(map_filename=map_1_file, 
                                      speed_profile_filename=spd_file, 
                                      closed_track=closed_loop)
    
    front_vehicle_motion_generator = preceding_vehicle_spd_profile_generation(horizon_length=8, time_interval=pv_dt)

    # Load A, B, C matrices required by perform_nonlinear_optimization_for_reward_tracking
    matrix_data_folder = os.path.join(parent_dir, "pv_scenario_generation_workspace", "data_driven_workspace")
    A, B, C = front_vehicle_motion_generator.load_matrices_from_file(data_folder_path=matrix_data_folder)
    if A is None or B is None or C is None:
        rospy.logerr(f"Failed to load reward-tracking model matrices from {matrix_data_folder}.")
        raise RuntimeError("Reward tracking model data is required for behavior generation.")

    traffic_map_manager.read_map_data()
    traffic_map_manager.read_speed_profile()
    
    dir_msg_publisher = rospy.Publisher('/runDirection', Int8, queue_size=2)
    lowlevel_heartbeat_publisher = rospy.Publisher('/low_level_heartbeat', Int8, queue_size=2)
    road_ref_pub = rospy.Publisher('/ref_traj_correction', ref_traj_correction, queue_size=2)

    msg_counter = 0
    ego_vehicle_pitch_from_acceleration = 0.0
    # start_t = time.time()
    prev_t = time.time()
    sim_t = 0.0
    ego_s_init = 0.0
    init_gap = 8.0
    reward_Q = 1.0
    reward_R = 10.0
    reward_R_du = 100.0
    reward_target = 20.0
    reward_tracking_duration = 10.0
    maintain_motion_duration = 5.0
    braking_jerk = -0.05
    braking_max_deceleration = -4.0

    while not rospy.is_shutdown():
        try:
            # Find time interval for current loop
            Dt = time.time() - prev_t
            # Refresh previous frame time
            prev_t = time.time()
            
            # Add traffic information to simulation managment class
            traffic_manager.serial_id = msg_counter
            virtual_traffic_sim_info_manager.update_ego_state(serial_id=traffic_manager.serial_id,
                                                              ego_x=traffic_manager.ego_x,
                                                              ego_y=traffic_manager.ego_y,
                                                              ego_z=traffic_manager.ego_z,
                                                              ego_yaw=traffic_manager.ego_yaw,
                                                              ego_pitch=traffic_manager.ego_pitch,
                                                              ego_v=traffic_manager.ego_v,
                                                              ego_acc=traffic_manager.ego_acc,
                                                              ego_omega=traffic_manager.ego_omega)
            
            s_ego_frenet, _ , _= traffic_map_manager.find_ego_vehicle_distance_reference(traffic_manager.ego_pose_ref)
            ego_vehicle_ref_poses = traffic_map_manager.find_traffic_vehicle_poses(s_ego_frenet, lane_id=0)
            
            # Initialize future states sequence
            front_s_t = [0.0] * 40
            front_v_t = [0.0] * 40
            front_a_t = [0.0] * 40
            
            run_dir_msg = Int8()
            run_dir_msg.data = run_direction
            dir_msg_publisher.publish(run_dir_msg)
            
            # Publish lowlevel heartbeat
            lowlevel_heartbeat_msg = Int8()
            lowlevel_heartbeat_msg.data = 1
            lowlevel_heartbeat_publisher.publish(lowlevel_heartbeat_msg)
            
            # Initialize the traffic simulation
            if (sim_t < 0.5 and traffic_manager.sim_start):
                # Update simulation time
                sim_t += Dt
                # Find initial distance as start distance on the map
                traffic_manager.traffic_initialization(s_ego=s_ego_frenet, ds=init_gap, line_number=0, vehicle_id=0, vehicle_id_in_lane=0)
                ego_s_init = s_ego_frenet
                continue
            else:
                # Start the scenario generation and simulation
                msg_counter += 1
                if traffic_manager.sim_start:
                    # Update simulation time
                    sim_t += Dt
                    
                    # Find front vehicle states for future horizon
                    front_vehicle_motion_generator.update_ego_vehicle_state(ego_a_t=traffic_manager.ego_acc,
                                                                            ego_v_t=traffic_manager.ego_v,
                                                                            ego_s_t=traffic_manager.ego_s,
                                                                            pv_a_t=traffic_manager.traffic_alon[0],
                                                                            pv_v_t=traffic_manager.traffic_v[0],
                                                                            pv_s_t=traffic_manager.traffic_s[0])
                    front_profile_s = None
                    front_profile_v = None
                    front_profile_a = None

                    if sim_t < reward_tracking_duration:
                        front_vehicle_motion_generator.perform_nonlinear_optimization_for_reward_tracking(
                            Q=reward_Q,
                            R=reward_R,
                            reward_target=reward_target,
                            a_max=3.0,
                            a_min=-3.0,
                            v_max=15.0,
                            v_min=0.0,
                            R_du=reward_R_du)

                        front_a_target = traffic_manager.ego_acc + float(front_vehicle_motion_generator.reward_tracking_u_opt[0])
                        front_a = np.clip(traffic_manager.traffic_alon[0], -3.0, 3.0)
                        traffic_manager.traffic_alon[0] = front_a + 0.05 * (front_a_target - front_a)
                    elif sim_t < reward_tracking_duration + maintain_motion_duration:
                        traffic_manager.traffic_alon[0] = 0.0
                    else:
                        front_profile_s, front_profile_v, front_profile_a = front_vehicle_motion_generator.generate_braking_profile(
                            jerk=braking_jerk,
                            max_deceleration=braking_max_deceleration)
                        if len(front_profile_a) > 1:
                            traffic_manager.traffic_alon[0] = front_profile_a[1]
                        else:
                            traffic_manager.traffic_alon[0] = 0.0

                    prev_front_v = traffic_manager.traffic_v[0]
                    traffic_manager.traffic_v[0] = np.clip(prev_front_v + traffic_manager.traffic_alon[0] * Dt, 0.0, 20.0)
                    traffic_manager.traffic_s[0] = traffic_manager.traffic_s[0] + prev_front_v * Dt + 0.5 * traffic_manager.traffic_alon[0] * Dt**2

                    if traffic_manager.traffic_v[0] <= 0.0:
                        traffic_manager.traffic_v[0] = 0.0
                        traffic_manager.traffic_alon[0] = 0.0

                    front_s_t[0] = round(traffic_manager.traffic_s[0], 3)
                    front_v_t[0] = round(traffic_manager.traffic_v[0], 3)
                    front_a_t[0] = round(traffic_manager.traffic_alon[0], 3)

                    for i in range(1, 20):
                        if front_profile_s is not None and i < len(front_profile_s):
                            front_s_t[i] = round(front_profile_s[i], 3)
                            front_v_t[i] = round(front_profile_v[i], 3)
                            front_a_t[i] = round(front_profile_a[i], 3)
                        elif not use_preview:
                            if front_v_t[i - 1] >= 20:
                                front_s_t[i] = round(front_s_t[i - 1] + front_v_t[i - 1] * pv_dt, 3)
                                front_a_t[i] = 0.0
                                front_v_t[i] = 20.0
                            elif front_v_t[i - 1] <= 0:
                                front_s_t[i] = front_s_t[i - 1]
                                front_a_t[i] = 0.0
                                front_v_t[i] = 0.0
                            else:
                                front_v_t[i] = round(np.clip(front_v_t[i - 1] + front_a_t[i - 1] * pv_dt, 0, 20), 3)
                                front_s_t[i] = round(front_s_t[i - 1] + front_v_t[i - 1] * pv_dt + 0.5 * front_a_t[i - 1] * pv_dt ** 2, 3)
                                front_a_t[i] = front_a_t[i - 1]
                        else:
                            sim_dt = i * pv_dt
                            v_t, s_t, a_t = traffic_map_manager.find_front_vehicle_predicted_state(dt=sim_dt, sim_t=sim_t)
                            front_s_t[i] = round(s_t + ego_s_init + init_gap, 3)
                            front_v_t[i] = round(v_t, 3)
                            front_a_t[i] = round(a_t, 3)

                    # Find virtual traffic global poses
                    for i in range(num_Sv):
                        traffic_vehicle_poses = traffic_map_manager.find_traffic_vehicle_poses(traffic_manager.traffic_s[i], lane_id=0)
                        ego_vehicle_pitch_from_acceleration = traffic_manager.ego_acceleration_pitch_update(pitch_max=2 / RAD_TO_DEGREE, 
                                                                                                            pitch_min=-2 / RAD_TO_DEGREE, 
                                                                                                            acc_max=4.0, 
                                                                                                            acc_min=-6.0)
                        ego_vehicle_poses = [traffic_manager.ego_x, traffic_manager.ego_y,
                                             ego_vehicle_ref_poses[2], traffic_manager.ego_yaw,
                                             ego_vehicle_ref_poses[4]]
                        
                        # Find ego vehicle pose on frenet coordinate
                        _, yaw_s, v_longitudinal, v_lateral = traffic_map_manager.find_ego_frenet_pose(ego_poses=traffic_manager.ego_pose_ref,
                                                                                                       ego_yaw=ego_vehicle_poses[3],
                                                                                                       vy=traffic_manager.ego_v_north,
                                                                                                       vx=traffic_manager.ego_v_east)
                        
                        traffic_manager.ego_vehicle_frenet_update(s=s_ego_frenet, l=0, sv=v_longitudinal, lv=v_lateral, yaw_s=yaw_s)
                        
                        local_traffic_vehicle_poses = host_vehicle_coordinate_transformation(traffic_vehicle_poses, ego_vehicle_poses)
                        
                        # Update virtual traffic braking status
                        traffic_manager.traffic_brake_status_update(vehicle_id=i)
                        virtual_vehicle_brake = traffic_manager.traffic_brake_status[i]
                        
                        # Update virtual traffic simulation information
                        virtual_traffic_sim_info_manager.update_virtual_vehicle_state(vehicle_id=i,
                                                                                   x=local_traffic_vehicle_poses[0],
                                                                                   y=-local_traffic_vehicle_poses[1], # Transfer to right hand coordinate
                                                                                   z=local_traffic_vehicle_poses[2],
                                                                                   yaw=-local_traffic_vehicle_poses[3], # Transfer to right hand coordinate
                                                                                   pitch=-local_traffic_vehicle_poses[4], # Transfer to right hand coordinate
                                                                                   acc=traffic_manager.traffic_alon[i],
                                                                                   vx=traffic_manager.traffic_v[i],
                                                                                   vy=0.0,
                                                                                   brake_status=virtual_vehicle_brake)

                else:
                    for i in range(num_Sv):
                        traffic_vehicle_poses = traffic_map_manager.find_traffic_vehicle_poses(traffic_manager.traffic_s[i] - s_ego_frenet, lane_id=0)
                        ego_vehicle_poses = [traffic_manager.ego_x, traffic_manager.ego_y,
                                             ego_vehicle_ref_poses[2], traffic_manager.ego_yaw,
                                             ego_vehicle_ref_poses[4]]
                        local_traffic_vehicle_poses = host_vehicle_coordinate_transformation(traffic_vehicle_poses, ego_vehicle_poses)
                        # Update virtual traffic braking status
                        if traffic_manager.traffic_alon[i] <= 0:
                            virtual_vehicle_brake = True
                        else:
                            virtual_vehicle_brake = False
                        # Update virtual traffic simulation information
                        virtual_traffic_sim_info_manager.update_virtual_vehicle_state(vehicle_id=i,
                                                                                   x=local_traffic_vehicle_poses[0],
                                                                                   y=-local_traffic_vehicle_poses[1], # Transfer to right hand coordinate
                                                                                   z=local_traffic_vehicle_poses[2],
                                                                                   yaw=-local_traffic_vehicle_poses[3], # Transfer to right hand coordinate
                                                                                   pitch=-local_traffic_vehicle_poses[4], # Transfer to right hand coordinate
                                                                                   acc=traffic_manager.traffic_alon[i],
                                                                                   vx=traffic_manager.traffic_v[i],
                                                                                   vy=0.0,
                                                                                   brake_status=virtual_vehicle_brake)
                    
            # Publish the traffic information
            # Construct the ROS message
            virtual_traffic_sim_info_manager.construct_hololens_info_msg()
            traffic_manager.construct_traffic_sim_info_msg(sim_t=sim_t)
            traffic_manager.construct_vehicle_state_sequence_msg(id=msg_counter, 
                                                                 t=sim_t, 
                                                                 s=front_s_t, 
                                                                 v=front_v_t, 
                                                                 a=front_a_t, 
                                                                 sim_start=traffic_manager.sim_start)

            # Publish the ROS message
            virtual_traffic_sim_info_manager.publish_virtual_sim_info()
            traffic_manager.publish_traffic_sim_info()
            traffic_manager.publish_vehicle_traj()

            # Publish road reference correction message
            road_ref_correction_msg = road_reference_correction_msg_prep(ego_vehicle_pitch_from_acceleration)
            road_ref_pub.publish(road_ref_correction_msg)
            
        except IndexError:
            print('Index error detected.')
        except RuntimeError:
            print('Runtime error detected.')

        rate.sleep()


if __name__ == "__main__":
    main_single_lane_following()
