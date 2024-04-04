#! /usr/bin/env python3

# This is a base class to traffic simulation

# Simulation script to run CMI traffic simulation
# Subscribe to "/bridge_to_lowlevel" for ego vehicle's kinematic and dynamic motion

import rospy
from std_msgs.msg import Float64MultiArray
from hololens_ros_communication.msg import hololens_info

import time
import numpy as np
import math

# Define constants
RAD_TO_DEGREE = 52.296

class CMI_traffic_sim:
    def __init__(self, num_vehicles, spd_profile, traj_map):
        self.map = traj_map
        self.spd_ref = spd_profile
        self.x = np.zeros((1,num_vehicles)).tolist()[0]
        self.y = np.zeros((1,num_vehicles)).tolist()[0]
        self.z = np.zeros((1,num_vehicles)).tolist()[0]
        self.yaw = np.zeros((1,num_vehicles)).tolist()[0]
        self.pitch = np.zeros((1,num_vehicles)).tolist()[0]
        self.alon = np.zeros((1,num_vehicles)).tolist()[0]
        self.vx = np.zeros((1,num_vehicles)).tolist()[0]
        self.vy = np.zeros((1,num_vehicles)).tolist()[0]
        self.omega = np.zeros((1,num_vehicles)).tolist()[0]
        self.msg_id = 0
        self.brake_status = np.zeros((1,num_vehicles)).tolist()[0]
        self.num_vehicles = 0
        self.Sv_id = np.zeros((1,num_vehicles)).tolist()[0]
        
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_z = 0.0
        self.ego_pitch = 0.0
        self.ego_yaw = 0.0
        self.ego_acc = 0.0
        self.ego_omega = 0.0
        self.ego_v = 0.0
        
        self.sub_lowlevel_bridge = rospy.Subscriber('/bridge_to_lowlevel', Float64MultiArray, self.lowlevel_bridge_callback)
        
        def lowlevel_bridge_callback(self, msg):
            self.ego_x = msg.data[0]
            self.ego_y = msg.data[1]
            self.ego_z = msg.data[2]
            self.ego_yaw = msg.data[5]
            self.ego_roll = msg.data[16]
            self.ego_pitch = msg.data[17] / RAD_TO_DEGREE
            self.ego_v_lon = msg.data[3]