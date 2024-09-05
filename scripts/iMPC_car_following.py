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
import casadi as ca
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
        
    def car_following_mpc(self, x_t0, front_vel, front_dist):
        # Initialize state variables and system control input
        opti = ca.Opti()
        x = opti.variable(3, self.h + 1)
        u = opti.variable(1, self.h)
        cost = 0.0
        
        # Add weights of MPC reference to the problem
        # P = np.diag(self.weight_list)
        
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
            # Add human driven vehicle motion model
            constr += [x[:, t + 1] == A @ x[:, t] + B @ u]
            # Add freeway speed cost
            cost += self.weight_list[0] * [x[1, t] - self.ref_list[0]] ** 2
            # Add desired acceleration cost
            cost += self.weight_list[1] * [x[2, t] - self.ref_list[1]] ** 2
            # Add relative velocity cost
            cost += self.weight_list[2] * [x[1, t] - front_vel[t] - self.ref_list[2]] ** 2
            # Add distance cost
            cost += self.weight_list[3] * [front_dist[t] - x[0, t]]
            # Add time headway cost
            cost += self.weight_list[4] * [[x[1, t] / [front_dist[t] - x[0, t]]] - self.ref_list[4]] ** 2
            # Add time to collision cost
            cost += self.weight_list[5] * [[x[1, t] - front_vel[t]] / [front_dist[t] - x[0, t]] - self.ref_list[5]] ** 2
            # Add control cost
            cost += self.weight_list[6] * u[0, t] ** 2
        return 0