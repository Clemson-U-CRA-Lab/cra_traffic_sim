#! /usr/bin/env python3
import numpy as np
import casadi
import sys
import time
import os
from matplotlib import pyplot as plt
from utils import *

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
        
        # Initialize A, B, C matrices (will be loaded via load_matrices_from_file)
        self.A = None
        self.B = None
        self.C = None
        
        # Initialize for storing optimal u
        self.reward_tracking_u_opt = None
        
    def load_matrices_from_file(self, data_folder_path=None):
        """Load A, B, C matrices from CSV files in the data_driven_workspace folder.
        
        Args:
            data_folder_path (str): Path to the data_driven_workspace folder. 
                                   If None, uses the default relative path.
        
        Returns:
            tuple: (A, B, C) - numpy arrays containing the loaded matrices
        """
        if data_folder_path is None:
            # Get the directory of the current script
            current_dirname = os.path.dirname(os.path.abspath(__file__))
            data_folder_path = os.path.join(current_dirname, 'data_driven_workspace')
        
        # Load matrices from CSV files
        try:
            A_file = os.path.join(data_folder_path, 'A_4x4_matrix.csv')
            B_file = os.path.join(data_folder_path, 'B_4x1_matrix.csv')
            C_file = os.path.join(data_folder_path, 'C_1x4_matrix.csv')
            
            self.A = np.loadtxt(A_file, delimiter=',')
            self.B = np.loadtxt(B_file, delimiter=',')
            self.C = np.loadtxt(C_file, delimiter=',')
            
            # Ensure proper shape for B (should be column vector)
            if self.B.ndim == 1:
                self.B = self.B.reshape(-1, 1)
            
            # Ensure proper shape for C (should be row vector)
            if self.C.ndim == 1:
                self.C = self.C.reshape(1, -1)
            
            print(f"Matrices loaded successfully from {data_folder_path}")
            print(f"A shape: {self.A.shape}, B shape: {self.B.shape}, C shape: {self.C.shape}")
            
            return self.A, self.B, self.C
        
        except FileNotFoundError as e:
            print(f"Error: Could not find matrix files in {data_folder_path}")
            print(f"Details: {e}")
            return None, None, None
        except Exception as e:
            print(f"Error loading matrices: {e}")
            return None, None, None

        
    def _koopman_lift_sindy(self, ds, dv, v_ego, a_ego):
        """Lift original state to Koopman space using the updated SINDy basis from MATLAB counterpart."""
        return np.array([ds, dv, ds**2, dv**2], dtype=float)

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
            cost += 100*((s[1, i] - self.ego_v[i]) - ttc_i_ref*(s[0, i] - self.ego_s[i]))**2
            # Add acceleration cost
            cost += 1*u[i-1]**2
            # Add slack variables
            cost += 1e6*(e_spd[i]**2 + e_ds[i]**2)
            
            # Define system dynamics
            opti.subject_to(s[0, i] == s[0, i-1] + s[1, i-1]*self.dT + 0.5*u[i-1]*self.dT**2)
            opti.subject_to(s[1, i] == s[1, i-1] + u[i-1]*self.dT)
            
            # Add safe distance constraint
            opti.subject_to(s[0, i] >= 5.0 + e_ds[i])
            
            # Add speed limit constraint
            opti.subject_to(s[1, i] <= v_max)
            opti.subject_to(s[1, i] >= v_min)
            
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
        
        # print("Solution found!")

    def perform_nonlinear_optimization_for_reward_tracking(self, Q, R, reward_target, a_max=None, a_min=None, v_max=None, v_min=None, du_max=None, du_min=None, R_du=0.0):
        """Optimize PV motion to track a target reward using lifted state-space model.

        Q: n x n state cost weight matrix (or scalar/diagonal vector)
        R: m x m control cost weight matrix (or scalar/diagonal vector)
        reward_target: scalar target reward value
        a_max: maximum acceleration (optional)
        a_min: minimum acceleration (optional)
        v_max: maximum velocity (optional)
        v_min: minimum velocity (optional)
        du_max: maximum change in control input between steps (optional)
        du_min: minimum change in control input between steps (optional)
        R_du: cost weight for rate of change in control input (default 0.0)
        """
        # Input checks
        if not isinstance(reward_target, (int, float)):
            raise ValueError("reward_target must be a scalar number.")
        if du_max is not None and not isinstance(du_max, (int, float)):
            raise ValueError("du_max must be numeric.")
        if du_min is not None and not isinstance(du_min, (int, float)):
            raise ValueError("du_min must be numeric.")
        if not isinstance(R_du, (int, float)) or R_du < 0:
            raise ValueError("R_du must be a non-negative scalar number.")

        if self.A is None or self.B is None or self.C is None:
            raise ValueError("Matrices A, B, C must be loaded before calling this method.")
        
        # Use stored matrices
        n = self.A.shape[0]
        if self.A.shape[1] != n:
            raise ValueError('A must be square (n x n)')
        m = self.B.shape[1]
        if self.B.shape[0] != n:
            raise ValueError('B must have n rows')
        if self.C.shape[1] != n:
            raise ValueError('C must have length n')

        # Standardize cost matrices
        Q = np.asarray(Q, dtype=float)
        if Q.ndim == 1:
            Q = np.diag(Q)
        elif Q.size == 1:
            Q = np.eye(n) * float(Q)
        if Q.shape != (n, n):
            raise ValueError('Q must be n x n, scalar, or n-vector')

        R = np.asarray(R, dtype=float)
        if R.ndim == 1:
            R = np.diag(R)
        elif R.size == 1:
            R = np.eye(m) * float(R)
        if R.shape != (m, m):
            raise ValueError('R must be m x m, scalar, or m-vector')

        # Initial lifted state from current states using Koopman SINDY lifting
        ds = self.pv_s - self.ego_s[0]
        dv = self.pv_v - self.ego_v[0]
        v_ego = self.ego_v[0]
        a_ego = self.ego_a[0]

        lift_z = self._koopman_lift_sindy(ds, dv, v_ego, a_ego)
        x0 = np.zeros(n)
        x0[: min(n, lift_z.size)] = lift_z[: min(n, lift_z.size)]

        # if n > lifted dimension, fill extra entries with current PV quantities
        if n > lift_z.size:
            extras = np.array([self.pv_s, self.pv_v, self.pv_a])
            x0[lift_z.size: min(n, lift_z.size + extras.size)] = extras[: min(n - lift_z.size, extras.size)]

        casA = casadi.DM(self.A)
        casB = casadi.DM(self.B)
        casC = casadi.DM(self.C)
        casQ = casadi.DM(Q)
        casR = casadi.DM(R)

        opti = casadi.Opti()
        x = opti.variable(n, self.h)
        u = opti.variable(m, self.h - 1)
        rw = opti.variable(self.h)
        e_r = opti.variable(self.h)

        # Initial condition
        opti.subject_to(x[:, 0] == casadi.DM(x0))
        opti.subject_to(e_r >= 0)

        cost = 0
        for i in range(1, self.h):
            # State-transition
            opti.subject_to(x[:, i] == casA @ x[:, i-1] + casB @ u[:, i-1])

            # Add constraints on control input (assuming u[0] is acceleration)
            if a_max is not None:
                opti.subject_to(u[0, i-1] <= a_max)
            if a_min is not None:
                opti.subject_to(u[0, i-1] >= a_min)

            # Add rate limit on control input (delta u)
            if i == 1:
                u_previous = self.pv_a - self.ego_a[0]  # current relative input
                if du_max is not None:
                    opti.subject_to(u[0, i-1] - u_previous <= du_max)
                if du_min is not None:
                    opti.subject_to(u[0, i-1] - u_previous >= du_min)
            else:
                if du_max is not None:
                    opti.subject_to(u[0, i-1] - u[0, i-2] <= du_max)
                if du_min is not None:
                    opti.subject_to(u[0, i-1] - u[0, i-2] >= du_min)

            # Add constraints on velocity (assuming x[1] is velocity if n >= 2)
            if n >= 2:
                if v_max is not None:
                    opti.subject_to(x[1, i] <= v_max)
                if v_min is not None:
                    opti.subject_to(x[1, i] >= v_min)

            # Reward and target tracking cost
            opti.subject_to(rw[i] == (casC @ x[:, i])[0])
            cost += 1 * (rw[i] - reward_target)**2

            # Regularization on state and input
            cost += casadi.mtimes([x[:, i].T, casQ, x[:, i]])
            cost += casadi.mtimes([u[:, i-1].T, casR, u[:, i-1]])

            # Cost on rate of change in control input
            if i >= 2:
                du = u[:, i-1] - u[:, i-2]
                cost += R_du * casadi.mtimes([du.T, du])

            # Relaxation margin to allow tracking feasibility
            cost += 1e6 * e_r[i]**2
            opti.subject_to(rw[i] >= reward_target - e_r[i])
            opti.subject_to(rw[i] <= reward_target + e_r[i])

        opti.minimize(cost)
        opti.solver('fatrop', {"expand": True, "print_time": 0}, {"print_level": 0})

        sol = opti.solve()

        self.reward_tracking_x_opt = sol.value(x)
        self.reward_tracking_u_opt = sol.value(u)
        self.reward_tracking_rw_opt = sol.value(rw)
        self.reward_tracking_err_opt = sol.value(e_r)

        return self.reward_tracking_x_opt, self.reward_tracking_u_opt, self.reward_tracking_rw_opt

if __name__ == "__main__":
    horizon_length = 6
    time_interval = 0.5
    v_max = 20.0
    v_min = 0.0
    a_max = 4.0
    a_min = -6.0
    
    front_v = 0.0
    front_a = 0.0
    front_s = 8.0
    
    ego_v = 0.0
    ego_a = 0.0
    ego_s = 0.0
    
    pv_spd_profile_gen = preceding_vehicle_spd_profile_generation(horizon_length, time_interval)
    pv_spd_profile_gen = preceding_vehicle_spd_profile_generation(horizon_length, time_interval)
    A, B, C = pv_spd_profile_gen.load_matrices_from_file()
    
    # Initialize nn_controller here if needed
    current_dirname = os.path.dirname(__file__)
    nn_pt_filename = current_dirname + '/traffic_following_control_dc_trained.pt'
    FCN_control = NN_controller(nn_pt_file=nn_pt_filename, input_num=3)
    
    sim_T = 0.0
    sim_end_T = 30.0
    dT = 0.1
    use_reward_tracking = True
    
    # Parameters for reward tracking
    Q = 1.0  # State cost weight
    R = 1.0  # Control cost weight
    R_du = 100.0  # Control rate change cost weight
    reward_target = 20.0  # Target reward value
    
    pv_a_log = [front_a]
    pv_v_log = [front_v]
    pv_s_log = [front_s]
    ego_a_log = [ego_a]
    ego_v_log = [ego_v]
    ego_s_log = [ego_s]
    a_gap_log = [front_a - ego_a]
    ttc_i_log = [(front_v - ego_v) / (front_s - ego_s)]
    ttc_i_ref_log = [0.1534 / (1 + np.exp(0.195 * (front_s - ego_s - 18.36)))]
    
    while sim_T < sim_end_T:
        sim_T += dT
        print("Simulation time:", round(sim_T, 2), "s", end='\r')
        ttc_i_ref = 0.1534 / (1 + np.exp(0.195 * (front_s - ego_s - 18.36)))
        ttc_i_ref_log.append(ttc_i_ref)
        
        # Generation front vehicle acceleration
        pv_spd_profile_gen.update_ego_vehicle_state(ego_a, ego_v, ego_s, front_a, front_v, front_s)
        if use_reward_tracking:
            pv_spd_profile_gen.perform_nonlinear_optimization_for_reward_tracking(Q, R, reward_target, a_max=3, a_min=-3, v_max=v_max, v_min=v_min, R_du=R_du)
            front_a_t = ego_a + pv_spd_profile_gen.reward_tracking_u_opt[0]
            front_a = front_a + 0.5 * (front_a_t - front_a)  # Add filter to front_a
        else:
            pv_spd_profile_gen.perform_nonlinear_optimization_for_pv_spd(ttc_i_ref, v_max=v_max, v_min=v_min, a_max=a_max, a_min=a_min)
            front_a = pv_spd_profile_gen.pv_a_opt[0]
        
        # Use NN controller to get ego vehicle acceleration
        ego_a_t = FCN_control.step_forward(s_vt=ego_v, pv_vt=front_v,
                                             s_st=ego_s, pv_st=front_s,
                                             s_at=ego_a, pv_at=front_a,
                                             use_prediction_horizon=True, sim_t=sim_T)
        
        # Add filter to ego_a
        ego_a = ego_a + 0.5 * (ego_a_t - ego_a)
        
        # Update vehicle states
        front_v = front_v + front_a * dT
        front_s = front_s + front_v * dT + 0.5 * front_a * dT**2
        ego_v = ego_v + ego_a * dT
        ego_s = ego_s + ego_v * dT + 0.5 * ego_a * dT**2
        
        # Log data
        pv_a_log.append(front_a)
        pv_v_log.append(front_v)
        pv_s_log.append(front_s)
        ego_a_log.append(ego_a)
        ego_v_log.append(ego_v)
        ego_s_log.append(ego_s)
        a_gap_log.append(front_a - ego_a)
        ttc_i_log.append((front_v - ego_v) / (front_s - ego_s))
        
    print("Simulation completed.")
    
    # Plot result in four subplots: ego speed & pv_speed vs time, distance gap vs time, ego acceleration & pv_acceleration vs time, ttc_i vs time
    time_log = np.arange(0, sim_end_T+dT, dT)
    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(time_log, pv_v_log, label='Preceding Vehicle Speed')
    plt.plot(time_log, ego_v_log, label='Ego Vehicle Speed')
    plt.ylabel('Speed (m/s)')
    plt.title('Preceding Vehicle and Ego Vehicle Speed Profiles')
    plt.legend()
    plt.grid()
    plt.subplot(4, 1, 2)
    distance_gap_log = np.array(pv_s_log) - np.array(ego_s_log)
    plt.plot(time_log, distance_gap_log, label='Distance Gap (PV - Ego)')
    plt.ylabel('Distance (m)')
    plt.title('Preceding Vehicle and Ego Vehicle Position Profiles')
    plt.legend()
    plt.grid()
    plt.subplot(4, 1, 3)
    plt.plot(time_log, pv_a_log, label='Preceding Vehicle Acceleration')
    plt.plot(time_log, ego_a_log, label='Ego Vehicle Acceleration')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Preceding Vehicle and Ego Vehicle Acceleration Profiles')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(time_log, a_gap_log, label='Acceleration Gap (front_a - ego_a)')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration Gap (m/s²)')
    plt.title('Preceding vs Ego Acceleration Gap')
    plt.legend()
    plt.grid()

    # Optionally keep TTCi in a separate figure
    plt.figure(figsize=(12, 4))
    plt.plot(time_log, ttc_i_log, label='TTCi')
    plt.plot(time_log, ttc_i_ref_log, label='TTCi Reference', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('TTCi (s)')
    plt.title('Time-To-Collision Index (TTCi) Profile')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    # Example usage
    # pv_spd_profile_gen.update_ego_vehicle_state(ego_a, ego_v, ego_s, front_a, front_v, front_s)
    # t = time.time()
    # pv_spd_profile_gen.perform_nonlinear_optimization_for_pv_spd(ttc_i_ref, v_max=v_max, v_min=v_min, a_max=a_max, a_min=a_min)
    # duration = time.time() - t
    # print("Optimization duration:", duration)
    
    # # Plot pv speed and ego speed and distance in two subplots
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.plot(pv_spd_profile_gen.pv_v_opt, label='Preceding Vehicle Speed')
    # plt.plot(pv_spd_profile_gen.ego_v, label='Ego Vehicle Speed')
    # plt.ylabel('Speed (m/s)')
    # plt.title('Preceding Vehicle and Ego Vehicle Speed Profiles')
    # plt.legend()
    
    # plt.subplot(2, 2, 2)
    # plt.plot(pv_spd_profile_gen.pv_s_opt - pv_spd_profile_gen.ego_s, label='Distance Gap (PV - Ego)')
    # plt.ylabel('Distance (m)')
    # plt.title('Preceding Vehicle and Ego Vehicle Position Profiles')
    # plt.legend()
    
    # plt.subplot(2, 2, 3)
    # plt.plot(pv_spd_profile_gen.pv_a_opt, label='Preceding Vehicle Acceleration')
    # plt.xlabel('Horizon Step')
    # plt.ylabel('Acceleration (m/s²)')
    # plt.title('Preceding Vehicle Acceleration Profile')
    
    # plt.subplot(2, 2, 4)
    # plt.plot(pv_spd_profile_gen.ttci_opt, label='TTCi')
    # plt.axhline(y=ttc_i_ref, color='r', linestyle='--', label='TTCi Reference')
    # plt.xlabel('Horizon Step')
    # plt.ylabel('TTCi (s)')
    # plt.title('Time-To-Collision Index (TTCi) Profile')
    
    # plt.show()