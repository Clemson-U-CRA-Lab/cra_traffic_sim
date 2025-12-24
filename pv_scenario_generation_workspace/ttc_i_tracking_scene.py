#! /usr/bin/env python3
import numpy as np
import casadi
import sys
import time

class preceding_vehicle_spd_profile_generation():
    def __init__(self, horizon_length, time_interval):
        self.h = horizon_length
        self.dT = time_interval
        self.ego_a = 0.0
        self.ego_v = 0.0
        self.ego_s = 0.0
        
        self.pv_a = 0.0
        self.pv_v = 0.0
        self.pv_s = 0.0
        
        self.da_opt = np.zeros(self.h)
        self.dv_opt = np.zeros(self.h)
        self.ds_opt = np.zeros(self.h)
        self.ttci_opt = np.zeros(self.h)
        
        self.ttc_i = 0.0
        
    def update_ego_vehicle_state(self, ego_a_t, ego_v_t, ego_s_t, pv_a_t, pv_v_t, pv_s_t):
        self.ego_a = ego_a_t
        self.ego_v = ego_v_t
        self.ego_s = ego_s_t
        
        self.pv_a = pv_a_t
        self.pv_v = pv_v_t
        self.pv_s = pv_s_t
        
        self.ttc_i = (self.pv_v - self.ego_v) / (self.pv_s - self.ego_s)
    
    def perform_nonlinear_optimization_for_pv_spd(self, ttc_i_ref, dv_tgt, dv_upper_lim, dv_lower_lim, da_upper_lim, da_lower_lim):
        # Update preceding vehicle state
        ds_t = self.pv_s - self.ego_s
        dv_t = self.pv_v - self.ego_v
        ttci_t = dv_t / ds_t
        
        # Create optimization problem
        opti = casadi.Opti()
        s = opti.variable(2, self.h)
        u = opti.variable(self.h - 1)
        e_spd = opti.variable(self.h)
        e_ds = opti.variable(self.h)
        s_0 = casadi.MX(np.matrix([ds_t, dv_t]))
        
        # Align initial state
        opti.subject_to(s[:, 0] == s_0.T)
        
        # Initialize cost
        cost = 0
        
        # Define dynamics
        for i in range(1, self.h):
            print(i)
            # Add ttci tracking cost
            cost += (s[1, i] - ttc_i_ref * s[0, i])**2
            # Add target speed tracking cost
            # cost += (s[1, i] - dv_tgt)**2
            # Add acceleration cost
            cost += 10*u[i-1]**2
            # Add slack variables
            cost += 1e8*(e_spd[i]**2 + e_ds[i]**2)
            
            # Define system dynamics
            opti.subject_to(s[0, i] == s[0, i-1] + s[1, i-1]*self.dT + 0.5*u[i-1]*self.dT**2)
            opti.subject_to(s[1, i] == s[1, i-1] + u[i-1]*self.dT)
            # opti.subject_to(s[2, i] == s[1, i] / s[0, i])
            
            # Add safe distance constraint
            opti.subject_to(s[0, i] >= 8.0 + e_ds[i])
            
            # Add speed limit constraint
            opti.subject_to(s[1, i] <= dv_upper_lim + e_spd[i])
            opti.subject_to(s[1, i] >= dv_lower_lim - e_spd[i])
            
            # Add acceleration limit constraint
            opti.subject_to(u[i-1] <= da_upper_lim)
            opti.subject_to(u[i-1] >= da_lower_lim)
        
        # Define minization problem
        opti.minimize(cost)
        
        # Define solver options
        opti.solver('fatrop', {"expand": False, "print_time": 1}, {"print_level": 1})
        
        sol = opti.solve()
        
        # Extract optimized values
        self.da_opt = sol.value(u)
        self.dv_opt = sol.value(s[1, :])
        self.ds_opt = sol.value(s[0, :])
        self.ttci_opt = self.dv_opt / self.ds_opt
        
        print("Solution found!")
        
if __name__ == "__main__":
    horizon_length = 10
    time_interval = 0.5
    v_max = 30.0
    v_min = 0.0
    a_max = 3.0
    a_min = -4.0
    
    front_v = 20.0
    front_a = 1.0
    front_s = 35.0
    
    ego_v = 11.0
    ego_a = -2.0
    ego_s = 0.0
    
    dv = front_v - ego_v
    da = front_a - ego_a
    ds = front_s - ego_s
    
    dv_upper_lim = v_max - front_v
    dv_lower_lim = v_min - front_v
    
    da_upper_lim = a_max - front_a
    da_lower_lim = a_min - front_a
    
    ttc_i_ref = -0.5
    dv_tgt = v_max - front_v
    
    pv_spd_profile_gen = preceding_vehicle_spd_profile_generation(horizon_length, time_interval)
    
    # Example usage
    pv_spd_profile_gen.update_ego_vehicle_state(ego_a, ego_v, ego_s, front_a, front_v, front_s)
    pv_spd_profile_gen.perform_nonlinear_optimization_for_pv_spd(ttc_i_ref, dv_tgt, dv_upper_lim, dv_lower_lim, da_upper_lim, da_lower_lim)
    
    print("Optimized acceleration:", pv_spd_profile_gen.da_opt)
    print("Optimized speed:", pv_spd_profile_gen.dv_opt)
    print("Optimized distance:", pv_spd_profile_gen.ds_opt)
    print("Optimized TTC:", pv_spd_profile_gen.ttci_opt)