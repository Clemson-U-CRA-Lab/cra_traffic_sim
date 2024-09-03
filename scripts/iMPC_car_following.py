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

class iMPC_tracker():
    def __init__(self, ref_list, weight_list):
        self.ref_list = ref_list
        self.weight_list = weight_list