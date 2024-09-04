#! /usr/bin/env python3

'''
For iMPC selected reference and weight, the candidates follows the order shown below
1. Freeway Velocity: V_h
2. Acceleration: a_h
3. Relative Velocity: V_r
4. Distance: d
5. Time headway: THWi
6. Time to collision: TTCi
7. Control input of HDV
'''

import rospy
from std_msgs.msg import Float64MultiArray
from hololens_ros_communication.msg import hololens_info

import time
import numpy as np
import math
import cvxpy as cp
import os

from sim_env_manager import *
from utils import *

# Define constants
RAD_TO_DEGREE = 52.296

class iMPC_tracker():
    def __init__(self, ref_list, weight_list, horizon_length, time_interval):
        self.ref_list = ref_list
        self.weight_list = weight_list
        self.h = horizon_length
        self.dt = time_interval
        
    def car_following_mpc(self, x_t0):
        # Initialize state variables and system control input
        x = cp.Variable((3, self.h + 1))
        u = cp.Variable((1, self.h))
        
        # Add weights of MPC reference to the problem
        P = np.diag(self.weight_list)
        
        # To-do: Add constraint to enforce initial state to the same as current measurement
        constr = [x[: 0] == x_t0]
        
        # To-do: Add constraint to vehicle states update
        A = np.array([[1, self.dt, self.dt ** 2 / 2], 
                      [0, 1, self.dt], 
                      [0, 0, 1]])
        
        B = np.array([[self.dt ** 3 / 6],
                      [self.dt ** 2 /  2],
                      [self.dt]])
        
        for t in range(self.h):
            constr += [x[:, t + 1] == A @ x[:, t] + B @ u]
        return 0