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

if __name__ == "__main__":
    # Path Parameters
    current_dirname = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dirname, os.pardir))

    num_Sv = rospy.get_param("/num_vehicles")

    map_1_filename = rospy.get_param("/map_1")
    map_2_filename = rospy.get_param("/map_2")
    spd_filename = rospy.get_param("/spd_map")

    map_1_file = os.path.join(parent_dir, "maps", map_1_filename)
    map_2_file = os.path.join(parent_dir, "maps", map_2_filename)
    spd_file = os.path.join(parent_dir, "speed_profile", spd_filename)

    rospy.init_node("CRA_Digital_Twin_Traffic")
    rate = rospy.Rate(100)

    traffic_manager = CMI_traffic_sim(num_vehicles=num_Sv)
    traffic_map_manager = cmi_road_reader(
        map_filename=map_1_file, speed_profile_filename=spd_file)
    traffic_map_manager.read_cmi_data()
    traffic_map_manager.read_speed_profile()

    msg_counter = 0
    start_t = time.time()

    while not rospy.is_shutdown():
        try:
            sim_t = time.time() - start_t
            msg_counter += 1
            if (sim_t < 0.5):
                s_ego_init = traffic_map_manager.find_ego_vehicle_distance_reference(np.array([[traffic_manager.ego_x],
                                                                                               [traffic_manager.ego_y],
                                                                                               [traffic_manager.ego_z]]))

                traffic_manager.traffic_initialization(s_ego=s_ego_init, ds=12)
                continue
            else:
                spd_t, dist_t, acc_t = traffic_map_manager.find_speed_profile_information(sim_t=sim_t)
                traffic_manager.traffic_update(dt=1/100, a=acc_t, v=spd_t, dist=dist_t, s_init=s_ego_init, vehicle_id=0, ds=12)
                # Find virtual traffic global poses
                for i in range(num_Sv):
                    traffic_vehicle_poses = traffic_map_manager.find_traffic_vehicle_poses(traffic_manager.traffic_s[i][0])
                    ego_vehicle_poses = [traffic_manager.ego_x, traffic_manager.ego_y,
                                         traffic_manager.ego_z, traffic_manager.ego_yaw,
                                         traffic_manager.ego_pitch]
                    local_traffic_vehicle_poses = host_vehicle_coordinate_transformation(traffic_vehicle_poses, ego_vehicle_poses)
                    # Todo  - Construct hololens message and publish as virtual_sim_info
            msg_counter += 1
            rate.sleep()
        except IndexError:
            continue
        except RuntimeError:
            continue
