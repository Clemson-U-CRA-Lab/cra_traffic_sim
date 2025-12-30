#! /usr/bin/env python3
import numpy as np
import casadi
import sys
import time
from matplotlib import pyplot as plt

class preceding_vehicle_spd_profile_generation():
    def __init__(self, horizon_length, time_interval):
        self.h = horizon_length
        self.dT = time_interval
        self.ego_a = np.zeros(self.h)
        self.ego_v = np.zeros(self.h)
        self.ego_s = np.zeros(self.h)
        
        self.pv_a = 0.0
        self.pv_v = 0.0
        self.pv_s = 0.0
        
        self.pv_a_opt = np.zeros(self.h)
        self.pv_v_opt = np.zeros(self.h)
        self.pv_s_opt = np.zeros(self.h)
        self.ttci_opt = np.zeros(self.h)
        
        self.ttc_i = 0.0
        
    def update_ego_vehicle_state(self, ego_a_t, ego_v_t, ego_s_t, pv_a_t, pv_v_t, pv_s_t):
        # Update ego vehicle state over the horizon
        for i in range(self.h):
            if i == 0:
                self.ego_a[i] = ego_a_t
                self.ego_v[i] = ego_v_t
                self.ego_s[i] = ego_s_t
            else:
                self.ego_a[i] = self.ego_a[i-1]
                self.ego_v[i] = self.ego_v[i-1] + self.ego_a[i-1]*self.dT
                self.ego_s[i] = self.ego_s[i-1] + self.ego_v[i-1]*self.dT + 0.5*self.ego_a[i-1]*self.dT**2
                
        # Update preceding vehicle state at current time   
        self.pv_a = pv_a_t
        self.pv_v = pv_v_t
        self.pv_s = pv_s_t
    
    def perform_nonlinear_optimization_for_pv_spd(self, ttc_i_ref, v_max, v_min, a_max, a_min):        
        # Create optimization problem
        opti = casadi.Opti()
        s = opti.variable(2, self.h)
        u = opti.variable(self.h - 1)
        e_spd = opti.variable(self.h)
        e_ds = opti.variable(self.h)
        s_0 = casadi.MX(np.matrix([self.pv_s, self.pv_v]))
        
        # Align initial state
        opti.subject_to(s[:, 0] == s_0.T)
        
        # Initialize cost
        cost = 0
        
        # Define dynamics
        for i in range(1, self.h):
            # Add ttci tracking cost
            cost += 10*((s[1, i] - self.ego_v[i]) - ttc_i_ref * (s[0, i] - self.ego_s[i]))**2
            # Add acceleration cost
            cost += u[i-1]**2
            # Add slack variables
            cost += 1e8*(e_spd[i]**2 + e_ds[i]**2)
            
            # Define system dynamics
            opti.subject_to(s[0, i] == s[0, i-1] + s[1, i-1]*self.dT + 0.5*u[i-1]*self.dT**2)
            opti.subject_to(s[1, i] == s[1, i-1] + u[i-1]*self.dT)
            
            # Add safe distance constraint
            opti.subject_to(s[0, i] >= 8.0 + e_ds[i])
            
            # Add speed limit constraint
            opti.subject_to(s[1, i] <= v_max + e_spd[i])
            opti.subject_to(s[1, i] >= v_min - e_spd[i])
            
            # Add acceleration limit constraint
            opti.subject_to(u[i-1] <= a_max)
            opti.subject_to(u[i-1] >= a_min)
        
        # Define minization problem
        opti.minimize(cost)
        
        # Define solver options
        opti.solver('fatrop', {"expand": True, "print_time": 0}, {"print_level": 0})
        
        sol = opti.solve()
        
        # Extract optimized values
        self.pv_a_opt = sol.value(u)
        self.pv_v_opt = sol.value(s[1, :])
        self.pv_s_opt = sol.value(s[0, :])
        self.ttci_opt = (self.pv_v_opt - self.ego_v) / (self.pv_s_opt - self.ego_s)
        
        print("Solution found!")
        
if __name__ == "__main__":
    horizon_length = 10
    time_interval = 0.5
    v_max = 15.0
    v_min = 0.0
    a_max = 3.0
    a_min = -4.0
    
    front_v = 14.0
    front_a = 0.0
    front_s = 75.0
    
    ego_v = 14.0
    ego_a = 0.0
    ego_s = 50.0
    
    ttc_i_ref = 0.25
    
    pv_spd_profile_gen = preceding_vehicle_spd_profile_generation(horizon_length, time_interval)
    
    # Example usage
    pv_spd_profile_gen.update_ego_vehicle_state(ego_a, ego_v, ego_s, front_a, front_v, front_s)
    t = time.time()
    pv_spd_profile_gen.perform_nonlinear_optimization_for_pv_spd(ttc_i_ref, v_max=v_max, v_min=v_min, a_max=a_max, a_min=a_min)
    duration = time.time() - t
    print("Optimization duration:", duration)
    
    # Plot pv speed and ego speed and distance in two subplots
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(pv_spd_profile_gen.pv_v_opt, label='Preceding Vehicle Speed')
    plt.plot(pv_spd_profile_gen.ego_v, label='Ego Vehicle Speed')
    plt.ylabel('Speed (m/s)')
    plt.title('Preceding Vehicle and Ego Vehicle Speed Profiles')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(pv_spd_profile_gen.pv_s_opt - pv_spd_profile_gen.ego_s, label='Distance Gap (PV - Ego)')
    plt.ylabel('Distance (m)')
    plt.title('Preceding Vehicle and Ego Vehicle Position Profiles')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(pv_spd_profile_gen.pv_a_opt, label='Preceding Vehicle Acceleration')
    plt.xlabel('Horizon Step')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Preceding Vehicle Acceleration Profile')
    
    plt.subplot(2, 2, 4)
    plt.plot(pv_spd_profile_gen.ttci_opt, label='TTCi')
    plt.axhline(y=ttc_i_ref, color='r', linestyle='--', label='TTCi Reference')
    plt.xlabel('Horizon Step')
    plt.ylabel('TTCi (s)')
    plt.title('Time-To-Collision Index (TTCi) Profile')
    
    plt.show()