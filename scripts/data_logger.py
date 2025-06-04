#! /usr/bin/env python3

# This code is designed to record ego vehicle data in simulation

import rospy
from std_msgs.msg import Float64MultiArray, Int8
from trajectory_msgs.msg import MultiDOFJointTrajectory
from nav_msgs.msg import Odometry
from hololens_ros_communication.msg import hololens_info
from cra_traffic_sim.msg import mpc_pose_reference
from nav_msgs.msg import Odometry
from vehicle_overtaking.msg import sas_pmp, overtaking_mpc

import time
import numpy as np
import math
import os
from utils import *

class data_logger:
    def __init__(self):
        self.sim_t = 0.0
        self.sim_t_prev = 0.0
        self.ros_t = 0.0
        self.ego_s = 0.0
        self.ego_v = 0.0
        self.ego_a = 0.0
        self.ego_adv_v = None
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.adv_v_sub = rospy.Subscriber('/sas_chatter', sas_pmp, self.adv_v_callback)
        self.vehicle_chatter_sub = rospy.Subscriber('/vehicle_chatter', overtaking_mpc, self.vehicle_chatter_callback)
        
    def odom_callback(self, msg):
        self.ego_s = round(msg.pose.pose.orientation.z, 2)
        spd_north = msg.twist.twist.linear.x
        spd_east = msg.twist.twist.linear.y
        self.ego_v = round((spd_north**2 + spd_east**2)**0.5, 2)
        self.ego_a = round(msg.pose.pose.orientation.x, 2)
    
    def adv_v_callback(self, msg):
        self.ego_adv_v = round(msg.des_vel, 2)
    
    def vehicle_chatter_callback(self, msg):
        self.sim_t = round(msg.Time_TL, 2)
        self.ros_t = msg.timestamp_n_x
        
    def update_logging_information(self):
        if self.sim_t != self.sim_t_prev:
            self.sim_t_prev = self.sim_t
            data = [self.sim_t, self.ros_t, self.ego_s, self.ego_v, self.ego_adv_v, self.ego_a]
        else:
            data = None
            
        return data
    
if __name__ == '__main__':
    rospy.init_node('data_logger')
    current_dirname = os.path.dirname(__file__)
    anl_data_logger = data_logger()
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        try:
            data = anl_data_logger.update_logging_information()
            if data is not None:
                with open(current_dirname + '/av_117_1.csv', 'a', newline='') as csvfile:
                    print(data)
                    writer = csv.writer(csvfile)
                    writer.writerow(data)
            rate.sleep()
        except IndexError:
            continue
        except RuntimeError:
            continue