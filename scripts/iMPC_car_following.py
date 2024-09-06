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
    def __init__(self, ref_list, weight_list, horizon_length, time_interval, speed_max, acc_max, acc_min, dist_min, ttci_max):
        self.ref_list = ref_list
        self.weight_list = weight_list
        self.h = horizon_length
        self.dt = time_interval
        self.v_max = speed_max
        self.a_max = acc_max
        self.a_min = acc_min
        self.d_min = dist_min
        self.ttci_max = ttci_max
        
    def car_following_mpc(self, x_t0, front_vel, front_dist):
        # Initialize state variables and system control input
        opti = ca.Opti()
        x = opti.variable(3, self.h + 1)
        u = opti.variable(1, self.h)
        x_0 = ca.MX(x_t0)
        cost = 0.0
        
        # Add weights of MPC reference to the problem
        # P = np.diag(self.weight_list)
        
        # To-do: Add constraint to enforce initial state to the same as current measurement
        opti.subject_to(x[:, 0] == x_0)
        
        # To-do: Add constraint to vehicle states update
        A = ca.MX(np.array([[1, self.dt, self.dt ** 2 / 2], 
                      [0, 1, self.dt], 
                      [0, 0, 1]]))
        
        B = ca.MX(np.array([[self.dt ** 3 / 6],
                      [self.dt ** 2 /  2],
                      [self.dt]]))
        
        for t in range(self.h):
            # Add human driven vehicle motion model
            opti.subject_to(x[:, t + 1] == A @ x[:, t] + B @ u[0, t])
            # Add acceleration limit constraint
            opti.subject_to(x[2, t] <= self.a_max)
            opti.subject_to(x[2, t] >= self.a_min)
            # Add speed limit constraint
            opti.subject_to(x[1, t] <= self.v_max)
            opti.subject_to(x[1, t] >= -0.5)
            # Add distance limit constraint
            opti.subject_to(front_dist[t] - x[0, t + 1] >= self.d_min)
            
            # Add freeway speed cost
            cost += self.weight_list[0] * (x[1, t] - self.ref_list[0]) ** 2
            # Add desired acceleration cost
            cost += self.weight_list[1] * (x[2, t] - self.ref_list[1]) ** 2
            # Add relative velocity cost
            cost += self.weight_list[2] * ((x[1, t] - front_vel[t]) - self.ref_list[2]) ** 2
            # Add distance cost
            cost += self.weight_list[3] * (front_dist[t] - x[0, t])
            # Add time headway cost
            cost += self.weight_list[4] * ((x[1, t] / (front_dist[t] - x[0, t])) - self.ref_list[4]) ** 2
            # Add time to collision cost
            cost += self.weight_list[5] * ((x[1, t] - front_vel[t]) / (front_dist[t] - x[0, t]) - self.ref_list[5]) ** 2
            # Add control cost
            cost += self.weight_list[6] * (u[0, t] ** 2)
        
        opti.minimize(cost)
        opti.solver('ipopt', {"print_time": False}, {"print_level": 0})#, {"acceptable_tol": 0.001})
        sol = opti.solve()
        
        s_pred = sol.value(x[0, :])
        v_pred = sol.value(x[1, :])
        a_pred = sol.value(x[2, :])
        u_pred = sol.value(u[0, :])
        
        return s_pred, v_pred, a_pred, u_pred
