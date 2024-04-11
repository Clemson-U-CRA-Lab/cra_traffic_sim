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

def road_map_building(filename, ros_msg_manager, ds):
    rospy.init_node("road_profile_mapping")
    rate = rospy.Rate(500)

    map_file = open(filename, "a")

    dist = 0
    pose = [ros_msg_manager.ego_x, ros_msg_manager.ego_y, ros_msg_manager.ego_z, ros_msg_manager.ego_yaw, ros_msg_manager.ego_pitch, dist]

    while not rospy.is_shutdown():
        try:
            # Check if distance between poses is larger than distance interval requirement
            dist_to_prev_pose = ((ros_msg_manager.ego_x - pose[0])**2 +
                                 (ros_msg_manager.ego_y - pose[1])**2 +
                                 (ros_msg_manager.ego_z - pose[2])**2)**0.5
            print(dist_to_prev_pose)
            dist += dist_to_prev_pose
            if (dist_to_prev_pose > ds):
                str_to_save = str(ros_msg_manager.ego_x) + "," + str(ros_msg_manager.ego_y) + "," + str(ros_msg_manager.ego_z) + "," + str(ros_msg_manager.ego_yaw) + "," + str(ros_msg_manager.ego_pitch) + "," + str(dist) + "\n"
                map_file.write(str_to_save)
                pose = [ros_msg_manager.ego_x, ros_msg_manager.ego_y, ros_msg_manager.ego_z, ros_msg_manager.ego_yaw, ros_msg_manager.ego_pitch, dist]
            rate.sleep()
        except RuntimeError:
            continue
        except IndexError:
            continue
    
    map_file.close()


if __name__ == "__main__":
    # Intialize ROS manager
    road_map_msg_manager = CMI_traffic_sim(num_vehicles=1)
    # Path Parameters
    current_dirname = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dirname, os.pardir))
    map_filename = rospy.get_param("/map")
    map_file = os.path.join(parent_dir, "maps", map_filename)
    distance_interval = rospy.get_param("/distance_gap")

    road_map_building(map_file, road_map_msg_manager, distance_interval)