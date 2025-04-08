#! /usr/bin/env python3

# This code is designed to track the planned trajectory from MIQP lane change traffic planner (Dr. Vahidi's research group)

import rospy
from std_msgs.msg import Float64MultiArray, Int8
from trajectory_msgs.msg import MultiDOFJointTrajectory
from nav_msgs.msg import Odometry
from hololens_ros_communication.msg import hololens_info
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
        self.real_time_planner_listener = True
        self.ego_s = 0.0
        self.ego_l = 0.0
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_ldot = 0.0
        self.ego_v = 0.0
        self.lane_change_end_s = 0.0
        self.x_origin = 0.0
        self.y_origin = 0.0
        self.gamma = 0.0
        self.LaneWidth = 0.0
        self.traj_s = np.array([0] * max_num_vehicles, dtype=float)
        self.traj_l = np.array([0] * max_num_vehicles, dtype=float)
        self.traj_s_vel = np.array([0] * max_num_vehicles, dtype=float)
        self.traj_l_vel = np.array([0] * max_num_vehicles, dtype=float)
        self.traj_acc = np.array([0] * max_num_vehicles, dtype=float)
        self.traj_lane_cmd = np.array([0] * max_num_vehicles, dtype=int)
        self.duration = np.array([0] * max_num_vehicles)
        self.traj_x = np.array([0] * max_num_vehicles, dtype=float)
        self.traj_y = np.array([0] * max_num_vehicles, dtype=float)
        self.traj_yaw = np.array([0] * max_num_vehicles, dtype=float)
        self.traj_s_store = np.array([0] * max_num_vehicles, dtype=float)
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
        self.ego_v = ((msg.twist.twist.linear.x)**2 + (msg.twist.twist.linear.y)**2)**0.5
        
    def lane_change_frenet_to_cartesian_conversion(self):
        # Find position of reference poses
        l_t = (self.traj_l - 1) * self.LaneWidth
        s_t = self.traj_s
        x = self.x_origin + s_t * math.cos(self.gamma) - l_t * math.sin(self.gamma)
        y = self.y_origin + s_t * math.sin(self.gamma) + l_t * math.cos(self.gamma)
        
        # Find orientation of reference poses
        dx = x[1:] - x[0:-1]
        dy = y[1:] - y[0:-1]
        theta = np.arctan2(dy, dx)
        
        # Update reference trajectory
        self.traj_x = x[1:]
        self.traj_y = y[1:]
        self.traj_yaw = theta
    
    def lane_change_trajectory_storage(self):
        # Locate the current lane ID
        l_t = (self.traj_l - 1) * self.LaneWidth
        l_ID = round(self.ego_l - 1)
        
        # Locate the lane change termination point
        lane_change_traj_ID = np.abs(l_ID - l_t)
        lane_change_end_ID = np.where(lane_change_traj_ID > 0.9)
        lane_change_end_ID = lane_change_end_ID[0][0]
        
        # Store the lane change path 
        s_t = self.traj_s
        x = self.x_origin + s_t * math.cos(self.gamma) - l_t * math.sin(self.gamma)
        y = self.y_origin + s_t * math.sin(self.gamma) + l_t * math.cos(self.gamma)
        
        # Find orientation of reference poses
        dx = x[1:] - x[0:-1]
        dy = y[1:] - y[0:-1]
        theta = np.arctan2(dy, dx)
        
        self.traj_x = x[1:]
        self.traj_y = y[1:]
        self.traj_yaw = theta
        self.traj_s_store = tuple(s_t)
        self.lane_change_end_s = s_t[lane_change_end_ID + 2]
        
        return s_t
    
    def lane_change_checker(self):
        # Check if lane change is needed
        lane_change_cmd = self.traj_lane_cmd
        lane_change_cmd[np.where(lane_change_cmd==0)] = 1
        
        # Add sliding windows of lane change determinator
        len_lane_change_det = round(np.clip(3 + 0.35 * self.ego_v, 3, 10))
        lane_change_needed = (np.std(lane_change_cmd[0: len_lane_change_det]) > 0.25)
        
        # Store lane change trajectory
        if lane_change_needed:
            self.real_time_planner_listener = False
            print('Lane change needed at s = ' + str(self.ego_s) + ' m. with horizon length of ' + str(len_lane_change_det))
            self.lane_change_trajectory_storage()
    
    def lane_change_end_checker(self):
        # Measure the distance between lane change end point
        dist_to_end = np.abs(self.ego_s - self.lane_change_end_s)
        
        # Release lane change trajectory
        if not self.real_time_planner_listener and dist_to_end < 0.25:
            print('Lane change ended at s = ' + str(self.ego_s) + ' m.')
            self.real_time_planner_listener = True
            dist_to_end_start_s = np.abs(np.array(self.traj_s_store) - self.ego_s)
            start_s_id = np.argmin(dist_to_end_start_s)
            traj_x = self.traj_x[start_s_id:]
            traj_y = self.traj_y[start_s_id:]
            traj_yaw = self.traj_yaw[start_s_id:]
            return traj_x, traj_y, traj_yaw
        else:
            # Organize the planned trajectory for reference
            # Find the nearest pose from stored trajectory
            dist_to_end_start_s = np.abs(np.array(self.traj_s_store) - self.ego_s)
            start_s_id = np.argmin(dist_to_end_start_s)
            traj_x = self.traj_x[start_s_id:]
            traj_y = self.traj_y[start_s_id:]
            traj_yaw = self.traj_yaw[start_s_id:]
            return traj_x, traj_y, traj_yaw
        
if __name__ == "__main__":
    # Initialize ros node
    rospy.init_node('highlevel_tracker_gen_path_debug')
    mpc_msg_publisher = rospy.Publisher('/lowlevel_mpc_reference_debug', mpc_pose_reference, queue_size=2)
    dir_msg_publisher = rospy.Publisher('/runDirection_debug', Int8, queue_size=2)
    lowlevel_heartbeat_publisher = rospy.Publisher('/low_level_heartbeat_debug', Int8, queue_size=2)
    rate = rospy.Rate(100)
    highlevel_msg_manager = anl_planner_listener(max_num_vehicles=30)
    
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
        
        highlevel_msg_manager.x_origin = x0
        highlevel_msg_manager.y_origin = y0
        highlevel_msg_manager.gamma = gamma
        highlevel_msg_manager.LaneWidth = 3.7
    elif run_direction == 1:
        x0 = lanes_xy[1][0]
        y0 = lanes_xy[1][1]
        x1 = lanes_xy[1][2]
        y1 = lanes_xy[1][3]
        gamma = math.atan2((y1-y0),(x1-x0))
        
        highlevel_msg_manager.x_origin = x0
        highlevel_msg_manager.y_origin = y0
        highlevel_msg_manager.gamma = gamma
        highlevel_msg_manager.LaneWidth = 3.7
    else:
        rospy.logfatal("Invalid Direction for run")
    
    # Message parameters
    msg_id = 0
    time_interval = 0.4
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
            
            if highlevel_msg_manager.highlevel_mpc_initiated:
                if highlevel_msg_manager.real_time_planner_listener:
                    highlevel_msg_manager.lane_change_checker()
                    highlevel_msg_manager.lane_change_frenet_to_cartesian_conversion()
                    # Prepare MPC reference message
                    mpc_referece_msg = construct_mpc_pose_reference_msg(serial_id=msg_id,
                                                                        num_ref=20,
                                                                        Dt=time_interval,
                                                                        ref_x=highlevel_msg_manager.traj_x.tolist(),
                                                                        ref_y=highlevel_msg_manager.traj_y.tolist(),
                                                                        ref_yaw=highlevel_msg_manager.traj_yaw.tolist(),
                                                                        ref_vel=highlevel_msg_manager.traj_s_vel.tolist(),
                                                                        ref_acc=highlevel_msg_manager.traj_acc.tolist())
                else:
                    traj_x, traj_y, traj_yaw = highlevel_msg_manager.lane_change_end_checker()
                    mpc_referece_msg = construct_mpc_pose_reference_msg(serial_id=msg_id,
                                                                        num_ref=20,
                                                                        Dt=time_interval,
                                                                        ref_x=traj_x.tolist(),
                                                                        ref_y=traj_y.tolist(),
                                                                        ref_yaw=traj_yaw.tolist(),
                                                                        ref_vel=highlevel_msg_manager.traj_s_vel.tolist(),
                                                                        ref_acc=highlevel_msg_manager.traj_acc.tolist())
                
                mpc_msg_publisher.publish(mpc_referece_msg)
                msg_id += 1
            rate.sleep()
        except IndexError:
            continue
        except RuntimeError:
            continue