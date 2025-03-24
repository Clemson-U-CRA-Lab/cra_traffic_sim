#! /usr/bin/env python3

# This code is designed to track the planned trajectory from MIQP lane change traffic planner (Dr. Vahidi's research group)

import rospy
from std_msgs.msg import Float64MultiArray, Int8
from trajectory_msgs.msg import MultiDOFJointTrajectory
from nav_msgs.msg import Odometry
from mach_e_control.msg import control_target

import time
import numpy as np
import math
import os
from sim_env_manager import *
from utils import *

# Define constants
RAD_TO_DEGREE = 52.296

class anl_sim_env:
    def __init__(self, time_interval, wheelbase):
        self.ego_s = 0.0
        self.ego_l = 0.0
        self.ego_yaw = 0.0
        self.ego_ldot = 0.0
        self.ego_sdot = 0.0
        self.steering = 0.0
        self.acc = 0.0
        self.ego_v = 0.0
        self.L = wheelbase
        self.dt = time_interval
        self.control_sub = rospy.Subscriber('/control_target_cmd', control_target, self.control_sub_callback)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
            
    def step_forward(self):
        self.ego_s = self.ego_s + self.ego_sdot * self.dt
        self.ego_l = self.ego_l + self.ego_ldot * self.dt
        self.ego_ldot = self.ego_v * math.sin(self.ego_yaw)
        self.ego_sdot = self.ego_v * math.cos(self.ego_yaw)
        self.ego_v = self.ego_v + self.acc * self.dt
        self.ego_yaw = self.ego_yaw + self.ego_v * math.tan(self.steering) / self.L * self.dt

    def control_sub_callback(self, msg):
        self.acc = msg.pedal_command
        curv = msg.steering_command
        self.steering = math.atan(self.L * curv)
    
    def pub_odom(self):
        odom_msg = Odometry()
        odom_msg.pose.pose.orientation.z = self.ego_s
        odom_msg.pose.pose.orientation.y = self.ego_l
        odom_msg.twist.twist.angular.y = self.ego_ldot
        self.odom_pub.publish(odom_msg)

if __name__ == "__main__":
    # Initialize ros node
    rospy.init_node('anl_sim')
    anl_sim = anl_sim_env(time_interval=0.1, wheelbase=4.9)
    rate = rospy.Rate(10)
    
    # Message parameters
    msg_id = 0
    start_t = time.time()

    while not rospy.is_shutdown():
        sim_t = time.time() - start_t
        try:
            anl_sim.pub_odom()
            rate.sleep()
        except IndexError:
            continue
        except RuntimeError:
            continue