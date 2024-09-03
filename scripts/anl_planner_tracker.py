#! /usr/bin/env python3

# This code is designed to track the planned trajectory from MIQP lane change traffic planner (Dr. Vahidi's research group)

import rospy
from std_msgs.msg import Float64MultiArray
from hololens_ros_communication.msg import hololens_info
from vehicle_overtaking.msg import overtaking_mpc

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
        self.serial_num = int(0)
        self.num_vehicle = 0
        self.S_id = np.zeros((1, 12), dtype=int).tolist()[0]
        self.S_x = np.zeros((1, 12)).tolist()[0]
        self.S_y = np.zeros((1, 12)).tolist()[0]
        # self.S_z = np.zeros((1,12)).tolist()[0]
        self.S_yaw = np.zeros((1, 12)).tolist()[0]
        self.V_x = np.zeros((1, 12)).tolist()[0]
        self.V_y = np.zeros((1, 12)).tolist()[0]
        self.A_x = np.zeros((1, 12)).tolist()[0]
        self.S_braking = np.zeros((1, 12)).tolist()[0]
        self.sub_anl_vehicle_chatter = rospy.Subscriber('/vehicle_chatter', overtaking_mpc, self.anl_vehicle_chatter_callback)
    
    def anl_vehicle_chatter_callback(self, msg):
        self.serial_num = msg.serial
        self.num_vehicle = msg.num_SVs_x
        self.S_id = msg.id_n_x
        self.S_x = msg.Sx_n_x
        self.S_y = msg.Sy_n_x
        self.S_z = msg.Sz_n_x
        self.S_yaw = msg.heading_n_x
        self.V_x = msg.v_n_x
        self.A_x = msg.a_n_x
        self.V_y = msg.Vy_n_x
        self.S_braking = msg.brake_n_x