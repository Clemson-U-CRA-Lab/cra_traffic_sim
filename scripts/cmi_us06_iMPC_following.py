#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from hololens_ros_communication.msg import hololens_info
from cra_traffic_sim.msg import iMPC

import time
import numpy as np
import math
import os

from sim_env_manager import *
from utils import *
from iMPC_car_following import iMPC_tracker

# Define constants
RAD_TO_DEGREE = 52.296

def main_us06_car_following():
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

    rospy.init_node("iMPC_car_following")
    rate = rospy.Rate(200)

    traffic_manager = CMI_traffic_sim(max_num_vehicles=12, num_vehicles=num_Sv)
    traffic_map_manager = road_reader(
        map_filename=map_1_file, speed_profile_filename=spd_file, closed_track=closed_loop)
    traffic_map_manager.read_map_data()
    traffic_map_manager.read_speed_profile()
    
    msg_id = 0
    start_t = time.time()
    sim_t = 0.0
    
    while not rospy.is_shutdown():
        try:
            # Find time interval for current loop
            Dt = time.time() - start_t - sim_t
            # Update simulation time
            sim_t = time.time() - start_t
            
        except IndexError:
            continue
        except RuntimeError:
            continue
return 0