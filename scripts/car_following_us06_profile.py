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
    map_2_filename = rospy.get_param("/map_2")
    spd_filename = rospy.get_param("/spd_map")

    map_1_file = os.path.join(parent_dir, "maps", map_1_filename)
    map_2_file = os.path.join(parent_dir, "maps", map_2_filename)
    spd_file = os.path.join(parent_dir, "speed_profile", spd_filename)

    rospy.init_node("CRA_Digital_Twin_Traffic")
    rate = rospy.Rate(100)

    traffic_manager = CMI_traffic_sim(num_vehicles=num_Sv)
    virtual_traffic_sim_info_manager = hololens_message_manager(num_vehicles=1, max_num_vehicles=12)
    traffic_map_manager = cmi_road_reader(
        map_filename=map_1_file, speed_profile_filename=spd_file, closed_track=closed_loop)
    traffic_map_manager.read_map_data()
    traffic_map_manager.read_speed_profile()

    msg_counter = 0
    start_t = time.time()

    while not rospy.is_shutdown():
        try:
            sim_t = time.time() - start_t
            
            virtual_traffic_sim_info_manager.serial = msg_counter
            virtual_traffic_sim_info_manager.Ego_acc = traffic_manager.ego_acc
            virtual_traffic_sim_info_manager.Ego_omega = traffic_manager.ego_omega
            virtual_traffic_sim_info_manager.Ego_v = traffic_manager.ego_v
            virtual_traffic_sim_info_manager.Ego_x = traffic_manager.ego_x
            virtual_traffic_sim_info_manager.Ego_y = traffic_manager.ego_y
            virtual_traffic_sim_info_manager.Ego_z = traffic_manager.ego_z
            virtual_traffic_sim_info_manager.Ego_yaw = traffic_manager.ego_yaw
            virtual_traffic_sim_info_manager.Ego_pitch = traffic_manager.ego_pitch
            
            if (sim_t < 0.5):
                # Find initial distance as start distance on the map
                s_ego_init, _ = traffic_map_manager.find_ego_vehicle_distance_reference(np.array([[traffic_manager.ego_x],
                                                                                               [traffic_manager.ego_y],
                                                                                               [traffic_manager.ego_z]]))

                traffic_manager.traffic_initialization(s_ego=s_ego_init, ds=12, line_number=0, vehicle_id=0, vehicle_id_in_lane=0)
                continue
            else:
                msg_counter += 1
                spd_t, dist_t, acc_t = traffic_map_manager.find_speed_profile_information(sim_t=sim_t)
                
                # Find virtual traffic global poses
                for i in range(num_Sv):
                    traffic_manager.traffic_update(dt=1/100, a=acc_t, v_tgt=spd_t, vehicle_id=i)
                    traffic_vehicle_poses = traffic_map_manager.find_traffic_vehicle_poses(traffic_manager.traffic_s[i][0])
                    ego_vehicle_poses = [traffic_manager.ego_x, traffic_manager.ego_y,
                                         traffic_manager.ego_z, traffic_manager.ego_yaw,
                                         traffic_manager.ego_pitch]

                    local_traffic_vehicle_poses = host_vehicle_coordinate_transformation(traffic_vehicle_poses, ego_vehicle_poses)
                    
                    # Update virtual traffic simulation information
                    virtual_traffic_sim_info_manager.virtual_vehicle_id[i] = i
                    virtual_traffic_sim_info_manager.S_v_x[i] = local_traffic_vehicle_poses[0]
                    virtual_traffic_sim_info_manager.S_v_y[i] = -local_traffic_vehicle_poses[1]
                    virtual_traffic_sim_info_manager.S_v_z[i] = local_traffic_vehicle_poses[2]
                    virtual_traffic_sim_info_manager.S_v_yaw[i] = -local_traffic_vehicle_poses[3]
                    virtual_traffic_sim_info_manager.S_v_pitch[i] = local_traffic_vehicle_poses[4]
                    
                    # Update virtual traffic braking status
                    if traffic_manager.traffic_alon[i] <= 0:
                        virtual_traffic_sim_info_manager.S_v_brake_status[i] = True
                    else:
                        virtual_traffic_sim_info_manager.S_v_brake_status[i] = False
                
                    virtual_traffic_sim_info_manager.S_v_acc[i] = traffic_manager.traffic_alon[i]
                    virtual_traffic_sim_info_manager.S_v_vx[i] = traffic_manager.traffic_v[i]
            
            # Publish the traffic information
            virtual_traffic_sim_info_manager.construct_hololens_info_msg()
            virtual_traffic_sim_info_manager.publish_virtual_sim_info()
            
            rate.sleep()
        except IndexError:
            continue
        except RuntimeError:
            continue


if __name__ == "__main__":
    main_single_lane_following()