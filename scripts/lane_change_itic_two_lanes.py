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


def main_lane_change_two_lanes():
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

    rospy.init_node("CRA_Digital_Twin_Traffic")
    rate = rospy.Rate(100)

    traffic_manager = CMI_traffic_sim(num_vehicles=num_Sv)
    virtual_traffic_sim_info_manager = hololens_message_manager(
        num_vehicles=num_Sv, max_num_vehicles=12)

    traffic_map_manager_0 = cmi_road_reader(
        map_filename=map_0_file, speed_profile_filename=spd_0_file, closed_track=closed_loop)
    traffic_map_manager_0.read_map_data()
    traffic_map_manager_0.read_speed_profile()

    traffic_map_manager_1 = cmi_road_reader(
        map_filename=map_1_file, speed_profile_filename=spd_1_file, closed_track=closed_loop)
    traffic_map_manager_1.read_map_data()
    traffic_map_manager_1.read_speed_profile()

    msg_counter = 0
    start_t = time.time()
    sim_t = 0

    while not rospy.is_shutdown():
        try:
            delta_t = time.time() - start_t - sim_t
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

            if sim_t < 0.5:
                # Find initial distance of two lanes
                # Lane 0
                s_ego_init_0, _ = traffic_map_manager_0.find_ego_vehicle_distance_reference(np.array([[traffic_manager.ego_x],
                                                                                                      [traffic_manager.ego_y],
                                                                                                      [traffic_manager.ego_z]]))
                # Lane 1
                s_ego_init_1, _ = traffic_map_manager_1.find_ego_vehicle_distance_reference(np.array([[traffic_manager.ego_x],
                                                                                                      [traffic_manager.ego_y],
                                                                                                      [traffic_manager.ego_z]]))
                
                traffic_manager.traffic_initialization(s_ego=s_ego_init_0, ds = 12, line_number=0, vehicle_id=0, vehicle_id_in_lane=0)
                traffic_manager.traffic_initialization(s_ego=s_ego_init_1, ds = 12, line_number=1, vehicle_id=1, vehicle_id_in_lane=0)
                continue
            else:
                msg_counter += 1
                spd_t_0, _, acc_t_0 = traffic_map_manager_0.find_speed_profile_information(sim_t=sim_t)
                spd_t_1, _, acc_t_1 = traffic_map_manager_1.find_speed_profile_information(sim_t=sim_t)
                
                # Find virtual traffic global poses
                for i in range(num_Sv):
                    # Check the vehicle id and its lane number
                    traffic_vehicle_lane_number = traffic_manager.traffic_l[i]
                    
                    if traffic_vehicle_lane_number == 0:
                        traffic_manager.traffic_update(dt=delta_t, a=acc_t_0, v_tgt=spd_t_0, vehicle_id=i, ds=12)
                        traffic_vehicle_poses = traffic_map_manager_0.find_traffic_vehicle_poses(traffic_manager.traffic_s[i][0])
                        
                    if traffic_vehicle_lane_number == 1:
                        traffic_manager.traffic_update(dt=delta_t, a=acc_t_1, v_tgt=spd_t_1, vehicle_id=i, ds=12)
                        traffic_vehicle_poses = traffic_map_manager_1.find_traffic_vehicle_poses(traffic_manager.traffic_s[i][0])

                    ego_vehicle_poses = [traffic_manager.ego_x, traffic_manager.ego_y,
                                         traffic_manager.ego_z, traffic_manager.ego_yaw,
                                         traffic_manager.ego_pitch]
                    
                    local_traffic_vehicle_poses = host_vehicle_coordinate_transformation(traffic_vehicle_poses, ego_vehicle_poses)
                    
                    # Update virtual traffic simulation information
                    virtual_traffic_sim_info_manager.virtual_vehicle_id[i] = i
                    virtual_traffic_sim_info_manager.S_v_x[i] = local_traffic_vehicle_poses[0]
                    virtual_traffic_sim_info_manager.S_v_y[i] = - local_traffic_vehicle_poses[1] # Convert into Unity Coordinate
                    virtual_traffic_sim_info_manager.S_v_z[i] = local_traffic_vehicle_poses[2]
                    virtual_traffic_sim_info_manager.S_v_yaw[i] = - local_traffic_vehicle_poses[3] # Convert into Unity Coordinate
                    virtual_traffic_sim_info_manager.S_v_pitch[i] = local_traffic_vehicle_poses[4]
                    
                    # Update virtual traffic braking status
                    if traffic_manager.traffic_alon[i] < 0:
                        virtual_traffic_sim_info_manager.S_v_brake_status[i] = True
                    else:
                        virtual_traffic_sim_info_manager.S_v_brake_status[i] = False
                
                    virtual_traffic_sim_info_manager.S_v_acc[i] = traffic_manager.traffic_alon[i]
                    virtual_traffic_sim_info_manager.S_v_vx[i] = traffic_manager.traffic_v[i]
            # Publish the traffic information
            virtual_traffic_sim_info_manager.construct_hololens_info_msg()
            virtual_traffic_sim_info_manager.publish_virtual_sim_info()
            
            rate.sleep()
            rate.sleep()
        except IndexError:
            continue
        except RuntimeError:
            continue


if __name__ == "__main__":
    main_lane_change_two_lanes()
