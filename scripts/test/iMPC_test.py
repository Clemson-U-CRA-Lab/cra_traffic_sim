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

import sys
import numpy as np
import os

sys.path.append("/home/tonyyaoyao/ANL_ws/src/cra_traffic_sim/scripts")
sys.path.append("/home/tonyyaoyao/ANL_ws/src/inverseMPC/scripts")

from sim_env_manager import *
from utils import *
from iMPC_car_following import iMPC_tracker
from matplotlib import pyplot as plt


# Define constants
RAD_TO_DEGREE = 52.296

def main():
    ref_list = [23.09, -0.011, 0.143, 70.678, 0.214, 0.0039, 0.006]
    weight_list = [1, 0.1, 0, 10, 10, 1, 0.1]
    horizon_length = 5
    time_interval = 0.2
    v_max = 50
    a_max = 10
    a_min = -10 
    d_min = 12
    ttci_max = 1
    
    car_following_control = iMPC_tracker(ref_list=ref_list, 
                                         weight_list=weight_list,
                                         horizon_length = horizon_length,
                                         time_interval = time_interval, 
                                         speed_max = v_max, 
                                         acc_max = a_max, 
                                         acc_min = a_min, 
                                         dist_min = d_min, 
                                         ttci_max = ttci_max)
    
    init_state = np.array([[0],[4],[-5]])
    v_front = np.array([0, 0, 0, 0, 0])
    d_front = np.array([20, 20, 20, 20, 20])
    s_pred, v_pred, a_pred, u_pred = car_following_control.car_following_mpc(init_state, v_front, d_front)
    print(s_pred)
    print(v_pred)
    print(a_pred)
    return 0
    
def main_US06_following():
    # Path Parameters
    current_dirname = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dirname, os.pardir))
    parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
    map_1_filename = "itic_dir0_lane1.csv"
    map_1_file = os.path.join(parent_dir, "maps", map_1_filename)
    spd_filename = "US06_CMI_Urban_speed_profile.csv"
    spd_file = os.path.join(parent_dir, "speed_profile", spd_filename)
    
    traffic_map_manager = road_reader(map_filename=map_1_file, speed_profile_filename=spd_file, closed_track=False)
    traffic_map_manager.read_map_data()
    traffic_map_manager.read_speed_profile()
    
    # Initialize iMPC tracker
    ref_list = [15.0, -0.025, -0.12, 9.00, 0.214, -0.004, 0.2]
    weight_list = [0.05, 0.1, 0.2, 0.001, 1, 1, 0.001]
    horizon_length = 8
    time_interval = 0.2
    v_max = 30
    a_max = 5
    a_min = -5
    d_min = 8
    ttci_max = 1
    
    car_following_control = iMPC_tracker(ref_list=ref_list, 
                                         weight_list=weight_list,
                                         horizon_length = horizon_length,
                                         time_interval = time_interval, 
                                         speed_max = v_max, 
                                         acc_max = a_max, 
                                         acc_min = a_min, 
                                         dist_min = d_min, 
                                         ttci_max = ttci_max)
    init_gap = 15.0
    t_start = 0
    t_end = 75
    t = t_start
    t_record = [t]
    dt = time_interval
    
    ego_init_state = np.array([[0.0],[0.0],[0.0]], dtype=float)
    front_state = np.array([[init_gap],[0.0],[0.0]], dtype=float)
    
    s_front = [init_gap]
    v_front = [0.0]
    
    s_ego = [0.0]
    v_ego = [0.0]
    
    step = 0
    
    # Simulate using selected driving cycle
    while t < t_end:
        v_front_mpc = np.zeros(horizon_length, dtype=float)
        s_front_mpc = np.zeros(horizon_length, dtype=float)
        
        # Select the front vehicle reference for the next few steps
        for i in range(horizon_length):
            t_ref = t + time_interval * (i + 1)
            speed_ref, dist_ref, _ = traffic_map_manager.find_speed_profile_information(sim_t=t_ref)
            v_front_mpc[i] = speed_ref
            s_front_mpc[i] = dist_ref + init_gap
        
        # Compute iMPC control output
        #try:
        s_pred, v_pred, a_pred, u_pred = car_following_control.car_following_mpc(ego_init_state, v_front_mpc, s_front_mpc)
        #except RuntimeError:
            #next
        
        # Update time step
        t += dt
        print('Simulation time: ', str(t), ego_init_state.T)
        t_record.append(t)
        
        # Update ego vehicle state
        # ego_init_state[2] = a_pred[0]
        # ego_init_state[1] += ego_init_state[2] * dt
        # ego_init_state[0] += ego_init_state[1] * dt
        
        ego_init_state[2] = a_pred[1]
        ego_init_state[1] = v_pred[1]
        ego_init_state[0] = s_pred[1]
        
        s_ego.append(ego_init_state[0, 0])
        v_ego.append(ego_init_state[1, 0])
        
        v_front_t, s_front_t, _ = traffic_map_manager.find_speed_profile_information(sim_t = t)
        s_front.append(s_front_t + init_gap)
        v_front.append(v_front_t)
        
        plt.figure(1)
        plt.xlim(-1, 1)
        plt.ylim(-10, 25)
        plt.scatter(x=[0], y=[0], c='r', s=350, marker='s')
        plt.scatter(x=[0], y=[-2], c='r', s=350, marker='s')
        plt.scatter(x=[0], y=[s_front[-1] - s_ego[-1]], c='g', s=350, marker='s')
        plt.scatter(x=[0], y=[s_front[-1] - s_ego[-1] + 2], c='g', s=350, marker='s')
        plt.plot([-0.5, -0.5], [-10, 35], '-k', linewidth=5)
        plt.plot([0.5, 0.5], [-10, 35], '-k', linewidth=5)
        plt.pause(0.001)
        # figname = 'CarFollowing_' + str(step) + ".png"
        # plt.savefig('/home/tonyyaoyao/Pictures/car_following/'+figname)
        plt.cla()
        
        step += 1
    
    # Plot the result to compare the speed
    plt.figure(2)
    plt.plot(t_record, v_ego)
    plt.plot(t_record, v_front)
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [m/s]')
    plt.legend(["Ego Vehicle", "Front Vehicle"])
    
    
    plt.figure(3)
    plt.plot(t_record, s_ego)
    plt.plot(t_record, s_front)
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.legend(["Ego Vehicle", "Front Vehicle"])
    
    plt.show()
    return 0


if __name__ == "__main__":
    #main()
    main_US06_following()