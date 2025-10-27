#! /usr/bin/env python3

# This code is designed to track the planned trajectory from MIQP lane change traffic planner (Dr. Vahidi's research group)

import rospy
from std_msgs.msg import Float64MultiArray, Int8
from trajectory_msgs.msg import JointTrajectory
from nav_msgs.msg import Odometry
from hololens_ros_communication.msg import hololens_info
from cra_traffic_sim.msg import mpc_pose_reference, a2b_cmd

import time
import numpy as np
import math
import os
from sim_env_manager import *
from utils import *

# Define constants
RAD_TO_DEGREE = 52.296

class anl_a2b_planner_listener:
    def __init__(self):
        self.serial_num = 0.0
        self.ego_s = 0.0
        self.ego_l = 0.0
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.tgt_acc = 0.0
        self.tgt_vel = 0.0
        self.driving_mode = 0.0
        self.sim_start_flag = False
        self.vo_joint_sub = rospy.Subscriber('/cf_out_joint', JointTrajectory, self.cf_out_joint_callback)
        self.bridge_highlevel_sub = rospy.Subscriber('/odom', Odometry, self.highlevel_bridge_callback)
    
    def cf_out_joint_callback(self, msg):
        self.serial_num = msg.header.seq
        self.tgt_vel = msg.points[0].velocities[0]
        self.tgt_acc = msg.points[0].accelerations[0]
        self.driving_mode = msg.points[0].effort[0]

        if self.serial_num > 0:
            self.sim_start_flag = True
            
    def highlevel_bridge_callback(self, msg):
        self.ego_s = msg.pose.pose.orientation.z
        self.ego_l = msg.pose.pose.orientation.y
        self.ego_x = msg.pose.pose.position.x
        self.ego_y = msg.pose.pose.position.y
        
if __name__ == "__main__":
    # Initialize ros node
    rospy.init_node('highlevel_a2b_tracker')
    mpc_msg_publisher = rospy.Publisher('/a2b_control_tgt', a2b_cmd, queue_size=2)
    dir_msg_publisher = rospy.Publisher('/runDirection', Int8, queue_size=2)
    lowlevel_heartbeat_publisher = rospy.Publisher('/low_level_heartbeat', Int8, queue_size=2)
    rate = rospy.Rate(100)
    highlevel_msg_manager = anl_a2b_planner_listener()
    
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
    time_interval = 1/20
    start_t = time.time()

    while not rospy.is_shutdown():
        sim_t = time.time() - start_t
        try:
            # Publish run direciton only once
            if sim_t < 1:
                run_dir_msg = Int8()
                run_dir_msg.data = run_direction
                dir_msg_publisher.publish(run_dir_msg)
            
            # Publish lowlevel heartbeat
            lowlevel_heartbeat_msg = Int8()
            lowlevel_heartbeat_msg.data = 1
            lowlevel_heartbeat_publisher.publish(lowlevel_heartbeat_msg)
            
            if highlevel_msg_manager.sim_start_flag:
                # Construct a2b command message
                a2b_cmd_msg = a2b_cmd()
                a2b_cmd_msg.serial = highlevel_msg_manager.serial_num
                a2b_cmd_msg.sim_t = highlevel_msg_manager.serial_num * time_interval
                # Fill reference poses
                a2b_cmd_msg.tgt_vel = highlevel_msg_manager.tgt_vel
                a2b_cmd_msg.tgt_acc = highlevel_msg_manager.tgt_acc
                a2b_cmd_msg.effort = highlevel_msg_manager.driving_mode
                
                # Publish a2b command message
                mpc_msg_publisher.publish(a2b_cmd_msg)
            rate.sleep()
        except IndexError:
            continue
        except RuntimeError:
            continue