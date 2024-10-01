#! /usr/bin/env python3

# This code is designed to track the planned trajectory from MIQP lane change traffic planner (Dr. Vahidi's research group)

import rospy
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import MultiDOFJointTrajectory
from nav_msgs.msg import Odometry
from hololens_ros_communication.msg import hololens_info
from vehicle_overtaking.msg import overtaking_mpc
from cra_traffic_sim.msg import mpc_pose_reference

import time
import numpy as np
import math
import os
from sim_env_manager import *
from utils import *

# Define constants
RAD_TO_DEGREE = 52.296

class anl_planner_listener:
    def __init__(self, max_num_vehicles):
        self.serial_num = int(0)
        self.num_traj_points = 0
        self.ego_s = 0.0
        self.ego_l = 0.0
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_ldot = 0.0
        self.traj_s = [0] * max_num_vehicles
        self.traj_l = [0] * max_num_vehicles
        self.traj_s_vel = [0] * max_num_vehicles
        self.traj_l_vel = [0] * max_num_vehicles
        self.traj_acc = [0] * max_num_vehicles
        self.traj_lane_cmd = [0] * max_num_vehicles
        self.duration = [0] * max_num_vehicles
        self.vo_joint_sub = rospy.Subscriber('/vo_joint', MultiDOFJointTrajectory, self.vo_joint_callback)
        self.bridge_highlevel_sub = rospy.Subscriber('/odom', Odometry, self.highlevel_bridge_callback)
    
    def vo_joint_callback(self, msg):
        self.serial_num = msg.header.seq
        self.num_traj_points = len(msg.points)
        
        for i in range(self.num_traj_points):
            self.traj_s[i] = msg.points[i].transforms[0].translation.x
            self.traj_l[i] = msg.points[i].transforms[0].translation.y
            self.traj_s_vel[i] = msg.points[i].velocities[0].linear.x
            self.traj_l_vel[i] = msg.points[i].velocities[0].linear.y
            self.traj_acc[i] = msg.points[i].accelerations[0].linear.x
            self.traj_lane_cmd[i] = msg.points[i].accelerations[0].linear.y
            self.duration[i] = msg.points[i].time_from_start.secs + msg.points[i].time_from_start.nsecs / 1**10
            
    def highlevel_bridge_callback(self, msg):
        self.ego_s = msg.pose.pose.orientation.z
        self.ego_l = msg.pose.pose.orientation.y
        self.ego_x = msg.pose.pose.position.x
        self.ego_y = msg.pose.pose.position.y