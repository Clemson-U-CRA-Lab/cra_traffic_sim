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
    def __init__(self, wheelbase, x_origin, y_origin, gamma, LaneWidth):
        self.ego_s = 0.0
        self.ego_l = 1.0
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
        
        self.x0 = x_origin
        self.y0 = y_origin
        self.gamma = gamma
        self.LaneWidth = LaneWidth
        
        self.control_sub = rospy.Subscriber('/control_target_cmd', control_target, self.control_sub_callback)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
            
    def step_forward(self, dt):
        self.ego_s = self.ego_s + self.ego_sdot * dt
        self.ego_l = self.ego_l + self.ego_ldot * dt
        self.ego_ldot = self.ego_v * math.sin(self.ego_s_yaw)
        self.ego_sdot = self.ego_v * math.cos(self.ego_s_yaw)
        self.ego_v = self.ego_v + self.acc * dt
        self.ego_s_yaw = self.ego_s_yaw + self.ego_v * math.tan(self.steering) / self.L * dt
        
        self.frenet_to_cartesian_coordinate_transform()
    
    def step_forward_kinematic_bicycle(self, dt):
        self.ego_s = self.ego_s + self.ego_v * math.cos(self.ego_s_yaw)* dt
        self.ego_l = self.ego_l + (self.ego_v * math.sin(self.ego_s_yaw) / self.LaneWidth) * dt
        self.ego_s_yaw = self.ego_s_yaw + (self.ego_v * math.tan(self.steering) / (self.L * self.LaneWidth)) * dt
        self.ego_v = self.ego_v + self.acc * dt
        if self.ego_v < 0:
            self.acc = 0.0
            self.ego_v = np.clip(self.ego_v, 0, 25)
        
        self.ego_ldot = self.ego_v * math.sin(self.ego_s_yaw) / self.LaneWidth
        self.ego_sdot = self.ego_v * math.cos(self.ego_s_yaw)
        
        self.frenet_to_cartesian_coordinate_transform()
        
    def step_forward_kinematic_bicycle_v2(self, dt):
        self. frenet_to_cartesian_coordinate_transform()
        
        self.ego_x = self.ego_x + self.ego_v * math.cos(self.ego_yaw)* dt
        self.ego_y = self.ego_y + (self.ego_v * math.sin(self.ego_yaw) / self.LaneWidth) * dt
        self.ego_yaw = self.ego_yaw + (self.ego_v * math.tan(self.steering) / (self.L)) * dt
        self.ego_v = self.ego_v + self.acc * dt
        
        self.ego_ldot = self.ego_v * math.sin(self.ego_s_yaw) / self.LaneWidth
        self.ego_sdot = self.ego_v * math.cos(self.ego_s_yaw)
        
        self.cartesian_to_frenet_coordinate_transform()

    def control_sub_callback(self, msg):
        self.acc = msg.pedal_command
        curv = msg.steering_command
        steering_cmd = math.atan(self.L * curv)
        self.steering = self.steering + 0.5 * (steering_cmd - self.steering)
        
    def pub_odom(self):
        odom_msg = Odometry()
        
        odom_msg.pose.pose.orientation.z = self.ego_s
        odom_msg.pose.pose.orientation.y = self.ego_l
        odom_msg.twist.twist.angular.y = self.ego_ldot
        odom_msg.twist.twist.linear.z = self.ego_s_yaw
        odom_msg.twist.twist.angular.z = self.ego_yaw
        odom_msg.twist.twist.linear.x = self.ego_v * math.cos(self.ego_yaw)
        odom_msg.twist.twist.linear.y = self.ego_v * math.sin(self.ego_yaw)
        odom_msg.pose.pose.position.x = self.ego_x
        odom_msg.pose.pose.position.y = self.ego_y
        odom_msg.pose.pose.orientation.x = self.acc
        
        self.odom_pub.publish(odom_msg)
    
    def cartesian_to_frenet_coordinate_transform(self):
        alpha = math.atan((self.ego_y - self.y0) / (self.ego_x - self.x0))
        beta = alpha - self.gamma
        dist_to_origin = ((self.ego_y - self.y0) ** 2 + (self.ego_x - self.x0) ** 2) ** 0.5
        self.ego_s = dist_to_origin * math.sin(beta)
        self.ego_l = dist_to_origin * math.cos(beta) / self.LaneWidth + 1
    
    def frenet_to_cartesian_coordinate_transform(self):
        # Find position of reference poses
        l_t = (self.ego_l - 1) * self.LaneWidth
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
    endpoint_file_path = os.path.join(os.path.dirname(__file__), "map_origins/laneEndpoints_long_itic.csv")
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
    anl_sim = anl_sim_env(wheelbase=4.0, x_origin=x0, y_origin=y0, gamma=gamma, LaneWidth=3.7)
    rate = rospy.Rate(50)
    
    # Message parameters
    msg_id = 0
    start_t = time.time()
    sim_t = 0

    while not rospy.is_shutdown():
        dt = time.time() - start_t - sim_t
        sim_t = time.time() - start_t
        try:
            print('Ego V: ' + str(round(anl_sim.ego_v, 2)) + ' m/s.')
            anl_sim.step_forward_kinematic_bicycle(dt=dt)
            anl_sim.pub_odom()
            rate.sleep()
        except IndexError:
            continue
        except RuntimeError:
            continue