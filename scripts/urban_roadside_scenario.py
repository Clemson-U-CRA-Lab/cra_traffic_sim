#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from hololens_ros_communication.msg import hololens_info

import time
import numpy as np
import math
import random
import os
from sim_env_manager import *
from utils import *

# Define constants
RAD_TO_DEGREE = 52.296


def main_urban_roadside_cut_in_scenario():
    # Path Parameters
    current_dirname = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dirname, os.pardir))

    num_Sv = rospy.get_param("/num_vehicles")
    track_style = rospy.get_param("/track_style")

    if track_style == "GrandPrix":
        closed_loop = True
    if track_style == "Rally":
        closed_loop = False

    map_0_filename = rospy.get_param("/map_1")
    map_1_filename = rospy.get_param("/map_2")
    spd_0_filename = rospy.get_param("/spd_map_1")
    spd_1_filename = rospy.get_param("/spd_map_2")

    map_0_file = os.path.join(parent_dir, "maps", map_0_filename)
    map_1_file = os.path.join(parent_dir, "maps", map_1_filename)
    spd_0_file = os.path.join(parent_dir, "speed_profile", spd_0_filename)
    spd_1_file = os.path.join(parent_dir, "speed_profile", spd_1_filename)

    rospy.init_node("CRA_urban_roadside_traffic_scene")
    rate = rospy.Rate(100)

    traffic_manager = CMI_traffic_sim(num_vehicles=num_Sv, max_num_vehicles=12)
    virtual_traffic_sim_info_manager = hololens_message_manager(
        num_vehicles=num_Sv, num_traffic_lights=0, max_num_vehicles=12, max_num_traffic_lights=10)

    traffic_map_manager_0 = road_reader(
        map_filename=map_0_file, speed_profile_filename=spd_0_file, closed_track=closed_loop)
    traffic_map_manager_0.read_map_data()
    traffic_map_manager_0.read_speed_profile()

    traffic_map_manager_1 = road_reader(
        map_filename=map_1_file, speed_profile_filename=spd_1_file, closed_track=closed_loop)
    traffic_map_manager_1.read_map_data()
    traffic_map_manager_1.read_speed_profile()

    msg_counter = 0
    start_t = time.time()
    sim_t = 0
    
    exit_vehicle_id = random.randint(1, num_Sv-1)
    print('Vehicle ' + str(exit_vehicle_id) + ' will merge into ego driving lane later.')
    spd_tgt = 5.0

    while not rospy.is_shutdown():
        try:
            dt = time.time() - start_t - sim_t
            sim_t = time.time() - start_t
            
            # Add ego vehicle's information to the hololens sender node
            virtual_traffic_sim_info_manager.serial = msg_counter
            virtual_traffic_sim_info_manager.Ego_acc = traffic_manager.ego_acc
            virtual_traffic_sim_info_manager.Ego_omega = traffic_manager.ego_omega
            virtual_traffic_sim_info_manager.Ego_v = traffic_manager.ego_v
            virtual_traffic_sim_info_manager.Ego_x = traffic_manager.ego_x
            virtual_traffic_sim_info_manager.Ego_y = traffic_manager.ego_y
            virtual_traffic_sim_info_manager.Ego_z = traffic_manager.ego_z
            virtual_traffic_sim_info_manager.Ego_yaw = traffic_manager.ego_yaw
            virtual_traffic_sim_info_manager.Ego_pitch = traffic_manager.ego_pitch

            if sim_t < 0.1:
                # Find initial distance of two lanes
                # Lane 0
                s_ego_init_0, _, _ = traffic_map_manager_0.find_ego_vehicle_distance_reference(np.array([[traffic_manager.ego_x],
                                                                                                      [traffic_manager.ego_y],
                                                                                                      [traffic_manager.ego_z]]))
                for i in range(num_Sv):
                    traffic_manager.traffic_initialization(s_ego=s_ego_init_0, ds = 10, line_number=0, vehicle_id=i, vehicle_id_in_lane=i)
                    [veh_x, veh_y, veh_z, veh_yaw, veh_pitch] = traffic_map_manager_0.find_traffic_vehicle_poses(traffic_manager.traffic_s[i])
                    traffic_vehicle_pose_ref = np.array([[veh_x], 
                                                         [veh_y], 
                                                         [veh_z]]) # Find road refernce from ego line
                    traffic_vehicle_ref_s, _, max_map_s = traffic_map_manager_1.find_ego_vehicle_distance_reference(traffic_vehicle_pose_ref)
                    traffic_vehicle_goal_pose = traffic_map_manager_1.find_traffic_vehicle_poses(dist_travelled=traffic_vehicle_ref_s)
                    veh_z = traffic_vehicle_goal_pose[2] # Use ego lane's vertial pose information
                    veh_pitch = traffic_vehicle_goal_pose[4] # Use ego lane's pitch information
                    traffic_manager.global_vehicle_update(veh_ID=i, x=veh_x, y=veh_y, z=veh_z, yaw=veh_yaw, pitch=veh_pitch)
                    if i == exit_vehicle_id:
                        exit_veh_control = stanley_vehicle_controller(x_init=veh_x, y_init=veh_y, z_init=veh_z, yaw_init=veh_yaw, pitch_init=veh_pitch)
                continue
            else:
                msg_counter += 1
                
                # Update ego vehicle poses
                ego_vehicle_poses = [traffic_manager.ego_x, traffic_manager.ego_y,
                                     traffic_manager.ego_z, traffic_manager.ego_yaw,
                                     traffic_manager.ego_pitch]
                
                # Find virtual traffic global poses
                for i in range(num_Sv):
                    if i == exit_vehicle_id:
                        # Find pure pursuit goal poses
                        traffic_vehicle_pose_ref = np.array([[exit_veh_control.x], 
                                                             [exit_veh_control.y], 
                                                             [exit_veh_control.z]])
                        ego_vehicle_pose_ref = np.array([[traffic_manager.ego_x],
                                                         [traffic_manager.ego_y],
                                                         [traffic_manager.ego_z]])
                        traffic_vehicle_ref_s, _, max_map_s = traffic_map_manager_1.find_ego_vehicle_distance_reference(traffic_vehicle_pose_ref)
                        traffic_vehicle_ref_s_0,_, _, = traffic_map_manager_0.find_ego_vehicle_distance_reference(traffic_vehicle_pose_ref)
                        ego_vehicle_ref_s, _, _ = traffic_map_manager_1.find_ego_vehicle_distance_reference(ego_vehicle_pose_ref)
                        
                        traffic_vehicle_ref_s += 3 # Add pure pursuit look ahead distance
                        
                        if traffic_vehicle_ref_s > max_map_s: # Map distance correction
                            traffic_vehicle_ref_s -= max_map_s
                        traffic_vehicle_goal_pose = traffic_map_manager_1.find_traffic_vehicle_poses(dist_travelled=traffic_vehicle_ref_s)
                        traffic_vehicle_pose_ref_0 = traffic_map_manager_0.find_traffic_vehicle_poses(dist_travelled=traffic_vehicle_ref_s_0)
                        traffic_vehicle_z = traffic_vehicle_pose_ref_0[2]
                        traffic_vehicle_pitch = traffic_vehicle_pose_ref_0[4]
                        exit_veh_control.pure_pursuit_controller(traffic_vehicle_goal_pose)
                        
                        # Start cut in when time to headway smaller than 2 seconds according to NHTSA safety design notes:
                        # OBJECTIVE TEST SCENARIOS FOR INTEGRATED VEHICLE-BASED SAFETY SYSTEMS
                        t_headway = (traffic_vehicle_ref_s - ego_vehicle_ref_s) / traffic_manager.ego_v
                        if t_headway <= 2 or exit_veh_control.v > 0.1:
                            if traffic_vehicle_ref_s < 540:
                                traffic_vehicle_acc = 2 * (spd_tgt - exit_veh_control.v)
                            else:
                                traffic_vehicle_acc = 2 * (0 - exit_veh_control.v)
                                
                        else:
                            traffic_vehicle_acc = 0.0
                        traffic_manager.traffic_alon[i] = exit_veh_control.acc
                        traffic_manager.traffic_v[i] = exit_veh_control.v
                        exit_veh_control.update_vehicle_state(acc=traffic_vehicle_acc, z=traffic_vehicle_z, pitch=traffic_vehicle_pitch, dt=dt)
                        traffic_vehicle_poses = exit_veh_control.get_traffic_pose()
                        local_traffic_vehicle_poses = host_vehicle_coordinate_transformation(traffic_vehicle_poses, ego_vehicle_poses)
                    else:
                        # Update traffic vehicle poses
                        traffic_vehicle_poses = [traffic_manager.traffic_x[i], traffic_manager.traffic_y[i], 
                                                 traffic_manager.traffic_z[i], traffic_manager.traffic_yaw[i],
                                                 traffic_manager.traffic_pitch[i]]

                        local_traffic_vehicle_poses = host_vehicle_coordinate_transformation(traffic_vehicle_poses, ego_vehicle_poses)
                    
                    # Update virtual traffic simulation information
                    virtual_traffic_sim_info_manager.virtual_vehicle_id[i] = i
                    virtual_traffic_sim_info_manager.S_v_x[i] = local_traffic_vehicle_poses[0]
                    # Transfer to right hand coordinate
                    virtual_traffic_sim_info_manager.S_v_y[i] = - local_traffic_vehicle_poses[1]
                    virtual_traffic_sim_info_manager.S_v_z[i] = local_traffic_vehicle_poses[2]
                    virtual_traffic_sim_info_manager.S_v_yaw[i] = - local_traffic_vehicle_poses[3]
                    virtual_traffic_sim_info_manager.S_v_pitch[i] = - local_traffic_vehicle_poses[4]
                    
                    # Update virtual traffic braking status
                    if traffic_manager.traffic_alon[i] < 0 or traffic_manager.traffic_v[i] == 0:
                        virtual_traffic_sim_info_manager.S_v_brake_status[i] = True
                        traffic_manager.traffic_brake_status[i] = True
                    else:
                        virtual_traffic_sim_info_manager.S_v_brake_status[i] = False
                        traffic_manager.traffic_brake_status[1] = False
                    
                    virtual_traffic_sim_info_manager.S_v_acc[i] = traffic_manager.traffic_alon[i]
                    virtual_traffic_sim_info_manager.S_v_vx[i] = traffic_manager.traffic_v[i]

            # Publish the traffic information
            virtual_traffic_sim_info_manager.construct_hololens_info_msg()
            traffic_manager.construct_traffic_sim_info_msg(sim_t=sim_t)
            virtual_traffic_sim_info_manager.publish_virtual_sim_info()
            traffic_manager.publish_traffic_sim_info()
            
            rate.sleep()
        except IndexError as e:
            print(e)
        except RuntimeError as e:
            print(e)


if __name__ == "__main__":
    main_urban_roadside_cut_in_scenario()
