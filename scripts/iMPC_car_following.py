#! /usr/bin/env python3

'''
For iMPC selected reference and weight, the candidates follows the order shown below
1. Freeway Velocity: V_h
2. Acceleration: a_h
3. Relative Velocity: V_r
4. Distance: d
5. Time headway: THWi
6. Time to collision: TTCi
7. Control input of HDV
'''

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

class iMPC_tracker():
    def __init__(self, ref_list, weight_list):
        self.ref_list = ref_list
        self.weight_list = weight_list
        
    def car_following_mpc():
        # To do - Use casadi to construct MPC solver for car following
        return 0