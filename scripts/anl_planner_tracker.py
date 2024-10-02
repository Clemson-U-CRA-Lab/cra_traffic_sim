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
        self.highlevel_mpc_initiated = False
        self.ego_s = 0.0
        self.ego_l = 0.0
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_ldot = 0.0
        self.traj_s = np.array([0] * max_num_vehicles)
        self.traj_l = np.array([0] * max_num_vehicles)
        self.traj_s_vel = np.array([0] * max_num_vehicles)
        self.traj_l_vel = np.array([0] * max_num_vehicles)
        self.traj_acc = np.array([0] * max_num_vehicles)
        self.traj_lane_cmd = np.array([0] * max_num_vehicles)
        self.duration = np.array([0] * max_num_vehicles)
        self.traj_x = np.array([0] * max_num_vehicles)
        self.traj_y = np.array([0] * max_num_vehicles)
        self.traj_yaw = np.array([0] * max_num_vehicles)
        self.vo_joint_sub = rospy.Subscriber('/vo_joint', MultiDOFJointTrajectory, self.vo_joint_callback)
        self.bridge_highlevel_sub = rospy.Subscriber('/odom', Odometry, self.highlevel_bridge_callback)
    
    def vo_joint_callback(self, msg):
        self.highlevel_mpc_initiated = True
        self.serial_num = msg.header.seq
        self.num_traj_points = len(msg.points)
        
        for i in range(self.num_traj_points):
            self.traj_s[i] = msg.points[i].transforms[0].translation.x
            self.traj_l[i] = msg.points[i].transforms[0].translation.y
            self.traj_s_vel[i] = msg.points[i].velocities[0].linear.x
            self.traj_l_vel[i] = msg.points[i].velocities[0].linear.y
            self.traj_acc[i] = msg.points[i].accelerations[0].linear.x
            self.traj_lane_cmd[i] = msg.points[i].accelerations[0].linear.y
            self.duration[i] = msg.points[i].time_from_start.secs + msg.points[i].time_from_start.nsecs / (10**9)
            
    def highlevel_bridge_callback(self, msg):
        self.ego_s = msg.pose.pose.orientation.z
        self.ego_l = msg.pose.pose.orientation.y
        self.ego_x = msg.pose.pose.position.x
        self.ego_y = msg.pose.pose.position.y
        
    def lane_change_frenet_to_cartesian_conversion(self, x_origin, y_origin, gamma, LaneWidth):
        # Find position of reference poses
        l_t = (self.traj_l - 1) * LaneWidth
        s_t = self.traj_s
        x = x_origin + s_t * math.cos(gamma) - l_t * math.sin(gamma)
        y = y_origin + s_t * math.sin(gamma) + l_t * math.cos(gamma)

        # Find orientation of reference poses
        dx = x[1:] - x[0:-1]
        dy = y[1:] - y[0:-1]
        theta = np.arctan2(dy, dx)
        
        # Update reference trajectory
        self.traj_x = x[1:]
        self.traj_y = y[1:]
        self.traj_yaw = theta

        #return x[1:], y[1:], theta
        
if __name__ == "__main__":
    # Initialize ros node
    rospy.init_node('highlevel_tracker')
    mpc_msg_publisher = rospy.Publisher('/lowlevel_mpc_reference', mpc_pose_reference, queue_size=2)
    rate = rospy.Rate(100)
    highlevel_msg_manager = anl_planner_listener(max_num_vehicles=30)
    
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
    
    # Message parameters
    msg_id = 0
    time_interval = 0.4
    
    while not rospy.is_shutdown():
        try:
            if highlevel_msg_manager.highlevel_mpc_initiated:
                # Update frenet pose to cartesian
                highlevel_msg_manager.lane_change_frenet_to_cartesian_conversion(x_origin=x0, y_origin=y0, gamma=gamma, LaneWidth=3.7)
                # Prepare MPC reference message
                mpc_referece_msg = construct_mpc_pose_reference_msg(serial_id=msg_id,
                                                                    num_ref=20,
                                                                    Dt=time_interval,
                                                                    ref_x=highlevel_msg_manager.traj_x.tolist(),
                                                                    ref_y=highlevel_msg_manager.traj_y.tolist(),
                                                                    ref_yaw=highlevel_msg_manager.traj_yaw.tolist(),
                                                                    ref_vel=highlevel_msg_manager.traj_s_vel.tolist(),
                                                                    ref_acc=highlevel_msg_manager.traj_acc.tolist())
                mpc_msg_publisher.publish(mpc_referece_msg)
                msg_id += 1
            rate.sleep()
        except IndexError:
            continue
        except RuntimeError:
            continue