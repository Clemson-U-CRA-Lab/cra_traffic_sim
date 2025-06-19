#! /usr/bin/env python3

# This code is designed to record ego vehicle data in simulation

import rospy
from datetime import datetime
from dataspeed_ulc_msgs.msg import UlcReport
from std_msgs.msg import Float64MultiArray
import argparse

import time
import numpy as np
import math
import os
from utils import *

class data_logger_mache_mr:
    def __init__(self):
        self.microsec_prev = 0.0
        self.microsec = 0.0
        self.sec = 0.0
        self.min = 0.0
        self.hour = 0.0
        self.ego_v = 0.0
        self.ego_a = 0.0
        self.sub_lowlevel_bridge = rospy.Subscriber('/bridge_to_lowlevel', Float64MultiArray, self.lowlevel_bridge_callback)
        
    def lowlevel_bridge_callback(self, msg):
        self.microsec = datetime.now().microsecond
        self.sec = datetime.now().second
        self.min = datetime.now().minute
        self.hour = datetime.now().hour
        self.ego_v = msg.data[15]
        self.ego_a = msg.data[6]
        
    def update_logging_information(self):
        if self.microsec != self.microsec_prev:
            self.microsec_prev = self.microsec
            print(self.microsec)
            data = [self.hour, self.min, self.sec, self.microsec, self.ego_v, self.ego_a]
        else:
            data = None
            
        return data
    
if __name__ == '__main__':
    rospy.init_node('data_logger_mr')
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=str, default="Tony", help="Driver for the test")
    parser.add_argument("--runID", type=str, default="0", help="Number of run for each driver")
    args = parser.parse_args()
    current_dirname = os.path.dirname(__file__)
    anl_data_logger = data_logger_mache_mr()
    rate = rospy.Rate(100)
    
    while not rospy.is_shutdown():
        try:
            data = anl_data_logger.update_logging_information()
            if data is not None:
                with open(current_dirname + '/' + args.driver + '_' + args.runID + '.csv', 'a', newline='') as csvfile:
                    print(data)
                    writer = csv.writer(csvfile)
                    writer.writerow(data)
            rate.sleep()
        except IndexError as e:
            print(e)
        except RuntimeError as e:
            print(e)