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
    def __init__(self, time_interval, wheelbase, x_origin, y_origin, gamma, LaneWidth):
        self.ego_s = 0.0
        self.ego_l = 0.0
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_yaw = 0.0
        self.ego_s_yaw = 0.0
        self.ego_ldot = 0.0
        self.ego_sdot = 0.0
        self.steering = 0.0
        self.acc = 0.0
        self.ego_v = 0.0
        self.L = wheelbase
        self.dt = time_interval
        
        self.x0 = x_origin
        self.y0 = y_origin
        self.gamma = gamma
        self.LaneWidth = LaneWidth
        
        self.control_sub = rospy.Subscriber('/control_target_cmd', control_target, self.control_sub_callback)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
            
    def step_forward(self):
        self.ego_s = self.ego_s + self.ego_sdot * self.dt
        self.ego_l = self.ego_l + self.ego_ldot * self.dt
        self.ego_ldot = self.ego_v * math.sin(self.ego_s_yaw)
        self.ego_sdot = self.ego_v * math.cos(self.ego_s_yaw)
        self.ego_v = self.ego_v + self.acc * self.dt
        self.ego_s_yaw = self.ego_s_yaw + self.ego_v * math.tan(self.steering) / self.L * self.dt
        
        self.frenet_to_cartesian_coordinate_transform()

    def control_sub_callback(self, msg):
        self.acc = msg.pedal_command
        curv = msg.steering_command
        self.steering = math.atan(self.L * curv)
    
    def pub_odom(self):
        odom_msg = Odometry()
        
        odom_msg.pose.pose.orientation.z = self.ego_s
        odom_msg.pose.pose.orientation.y = self.ego_l
        odom_msg.twist.twist.angular.y = self.ego_ldot
        odom_msg.twist.twist.linear.z = self.ego_s_yaw
        odom_msg.twist.twist.angular.z = self.ego_yaw
        odom_msg.pose.pose.position.x = self.ego_x
        odom_msg.pose.pose.position.y = self.ego_y
        odom_msg.pose.pose.orientation.x = self.acc
        self.odom_pub.publish(odom_msg)
    
    def frenet_to_cartesian_coordinate_transform(self):
        # Find position of reference poses
        l_t = self.ego_l * self.LaneWidth
        s_t = self.ego_s
        x = self.x0 + s_t * math.cos(self.gamma) - l_t * math.sin(self.gamma)
        y = self.y0 + s_t * math.sin(self.gamma) + l_t * math.cos(self.gamma)
        
        # Find orientation of reference poses
        theta = self.gamma + self.ego_s_yaw
        
        # Update reference trajectory
        self.ego_x = x
        self.ego_y = y
        self.ego_yaw = theta

if __name__ == "__main__":
    # Define map origins
    run_direction = rospy.get_param("/runDirection")
    endpoint_file_path = os.path.join(os.path.dirname(__file__), "map_origins/laneEndpoints_itic.csv")
    file = open(endpoint_file_path)
    lanes_xy = np.float_(list(csv.reader(file,delimiter=",")))
    file.close()
    
    if run_direction == 0:
        x0 = lanes_xy[0][0]
        y0 = lanes_xy[0][1]
        x1 = lanes_xy[0][2]
        y1 = lanes_xy[0][3]
        gamma = math.atan2((y1-y0),(x1-x0))
    elif run_direction == 1:
        x0 = lanes_xy[1][0]
        y0 = lanes_xy[1][1]
        x1 = lanes_xy[1][2]
        y1 = lanes_xy[1][3]
        gamma = math.atan2((y1-y0),(x1-x0))
    else:
        rospy.logfatal("Invalid Direction for run")
        quit()
        
    # Initialize ros node
    rospy.init_node('anl_sim')
    anl_sim = anl_sim_env(time_interval=0.1, wheelbase=4.9, x_origin=x0, y_origin=y0, gamma=gamma, LaneWidth=3.7)
    rate = rospy.Rate(10)
    
    # Message parameters
    msg_id = 0
    start_t = time.time()

    while not rospy.is_shutdown():
        sim_t = time.time() - start_t
        try:
            anl_sim.step_forward()
            anl_sim.pub_odom()
            rate.sleep()
        except IndexError:
            continue
        except RuntimeError:
            continue