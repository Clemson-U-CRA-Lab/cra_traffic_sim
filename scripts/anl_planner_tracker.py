#! /usr/bin/env python3

# This code is designed to track the planned trajectory from MIQP lane change traffic planner (Dr. Vahidi's research group)

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

class anl_planner_listener:
    def __init__(self, w_s_x, w_s_y, w_s_aw, w_s_vel, w_s_acc, w_u_acc_dot, w_u_steering):
        self.ref_s = 0.0 # MIQP predicted longitudinal state
        self.ref_l = 0.0 # MIQP predicted lateral state
        