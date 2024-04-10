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

    map_1_filename = rospy.get_param("/map_1")
    map_2_filename = rospy.get_param("/map_2")
    spd_1_filename = rospy.get_param("/spd_map_1")
    spd_2_filename = rospy.get_param("/spd_map_2")

    map_1_file = os.path.join(parent_dir, "maps", map_1_filename)
    map_2_file = os.path.join(parent_dir, "maps", map_2_filename)
    spd_1_file = os.path.join(parent_dir, "speed_profile", spd_1_filename)
    spd_2_file = os.path.join(parent_dir, "speed_profile", spd_2_filename)

    rospy.init_node("CRA_Digital_Twin_Traffic")
    rate = rospy.Rate(100)

    traffic_manager = CMI_traffic_sim(num_vehicles=num_Sv)
    virtual_traffic_sim_info_manager = hololens_message_manager(num_vehicles=12)
    
    traffic_map_manager_1 = cmi_road_reader(map_filename=map_1_file, speed_profile_filename=spd_1_file)
    traffic_map_manager_1.read_map_data()
    traffic_map_manager_1.read_speed_profile()
    
    traffic_map_manager_2 = cmi_road_reader(map_filename=map_2_file, speed_profile_filename=spd_2_file)
    traffic_map_manager_2.read_map_data()
    traffic_map_manager_2.read_speed_profile()

    msg_counter = 0
    start_t = time.time()


if __name__ == "__main__":
    main_lane_change_two_lanes()