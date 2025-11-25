#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from hololens_ros_communication.msg import hololens_info

import time
import numpy as np
import math
import os
from sim_env_manager import *
from utils import *

# Define constants
RAD_TO_DEGREE = 52.296


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

    map_1_filename = rospy.get_param("/map_1")
    spd_filename = rospy.get_param("/spd_map")
    run_sim = bool(rospy.get_param("/run_sim"))
    pv_dt = float(rospy.get_param("/pv_states_dt"))
    use_preview = bool(rospy.get_param("/use_preview"))

    map_1_file = os.path.join(parent_dir, "maps", map_1_filename)
    spd_file = os.path.join(parent_dir, "speed_profile", spd_filename)
    print('PV speed pfofile is: ' , spd_file)

    rospy.init_node("CRA_Digital_Twin_Traffic")
    rate = rospy.Rate(200)

    traffic_manager = CMI_traffic_sim(max_num_vehicles=12, num_vehicles=num_Sv, sil_simulation=run_sim)
    virtual_traffic_sim_info_manager = hololens_message_manager(num_vehicles=1, max_num_vehicles=200, max_num_traffic_lights=12, num_traffic_lights=0)
    traffic_map_manager = road_reader(map_filename=map_1_file, speed_profile_filename=spd_file, closed_track=closed_loop)
    traffic_map_manager.read_map_data()
    traffic_map_manager.read_speed_profile()

    msg_counter = 0
    # start_t = time.time()
    prev_t = time.time()
    sim_t = 0.0
    init_gap = 8.0

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
            ego_vehicle_ref_poses = traffic_map_manager.find_traffic_vehicle_poses(s_ego_frenet)
            
            # Initialize future states sequence
            front_s_t = [0.0] * 40
            front_v_t = [0.0] * 40
            front_a_t = [0.0] * 40
            
            # Initialize the traffic simulation
            if (sim_t < 0.5 and traffic_manager.sim_start):
                # Update simulation time
                sim_t += Dt
                # Find initial distance as start distance on the map
                traffic_manager.traffic_initialization(s_ego=s_ego_frenet, ds=init_gap, line_number=0, vehicle_id=0, vehicle_id_in_lane=0) # One vehicle in lane 0
                continue
            else:
                msg_counter += 1
                if traffic_manager.sim_start:
                    # Update simulation time
                    sim_t += Dt
                    spd_t, _, acc_t = traffic_map_manager.find_speed_profile_information(sim_t=sim_t)
                    if not use_preview:
                        front_s_t[0] = round(traffic_manager.traffic_s[0], 3)
                        front_v_t[0] = round(traffic_manager.traffic_v[0], 3)
                        front_a_t[0] = traffic_manager.traffic_alon[0]

                    # Find the states for next few time steps
                    for i in range(1, 20):
                        if not use_preview:
                            if front_v_t[i - 1] >= 30:
                                front_a_t[i] = 0
                                front_v_t[i] = 30
                            
                            if front_v_t[i - 1] <= 0:
                                front_a_t[i] = 0
                                front_v_t[i] = 0
                            
                            front_s_t[i] = round(front_s_t[i - 1] + front_v_t[i - 1] * pv_dt + 0.5 * front_a_t[i - 1] * pv_dt ** 2, 3)
                            front_v_t[i] = round(np.clip(front_v_t[i - 1] + front_a_t[i - 1] * pv_dt, 0, 20), 3)
                            front_a_t[i] = front_a_t[i - 1]
                        else:
                            sim_dt = i * pv_dt
                            v_t, s_t, a_t = traffic_map_manager.find_front_vehicle_predicted_state(dt=sim_dt, sim_t=sim_t)
                            front_s_t[i] = round(s_t + s_ego_frenet + init_gap, 3)
                            front_v_t[i] = round(v_t, 3)
                            front_a_t[i] = round(a_t, 3)

                    # Find virtual traffic global poses
                    for i in range(num_Sv):
                        traffic_manager.traffic_update(dt=Dt, a=acc_t, v_tgt=spd_t, vehicle_id=i)
                        traffic_vehicle_poses = traffic_map_manager.find_traffic_vehicle_poses(traffic_manager.traffic_s[i])
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
                else: # Simulation not started yet
                    for i in range(num_Sv):
                        traffic_vehicle_poses = traffic_map_manager.find_traffic_vehicle_poses(traffic_manager.traffic_s[i])
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

            
        except IndexError:
            print('Index error detected.')
        except RuntimeError:
            print('Runtime error detected.')

        rate.sleep()


if __name__ == "__main__":
    main_single_lane_following()
