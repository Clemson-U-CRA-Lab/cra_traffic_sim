"""Koopman reward-tracking optimizer restored from the historical runtime."""
import os
import re

import casadi
import numpy as np

class preceding_vehicle_spd_profile_generation():
    def __init__(self, horizon_length, time_interval, solver="ipopt"):
        self.solver = solver
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
        self.koopman_lift_method = None
        self.edmd_sigma = 0.5
        self.edmd_ranges = {
            "ds": (5.0, 50.0),
            "dv": (-10.0, 10.0),
            "ttci": (-0.5, 0.5),
            "thwi": (0.0, 2.5),
        }
        self.edmd_centers = None

        # Initialize for storing optimal u
        self.reward_tracking_u_opt = None
        self.reward_tracking_x_opt = None
        self.reward_tracking_rw_opt = None
        self.reward_tracking_err_opt = None
        self.reward_tracking_acc_err_opt = None
        self.reward_tracking_target_window = None

    def _infer_lift_configuration(self, state_dim):
        if state_dim == 4:
            self.koopman_lift_method = "sindy_baseline"
            self.edmd_centers = None
            return
        if state_dim == 6:
            self.koopman_lift_method = "sindy_ttci_thwi"
            self.edmd_centers = None
            return

        edmd_grid_count = round(state_dim ** 0.25)
        if edmd_grid_count ** 4 != state_dim:
            raise ValueError(
                f"Unsupported Koopman state dimension {state_dim}. "
                "Expected 4, 6, or an EDMD grid with equal centers per feature."
            )

        self.koopman_lift_method = "edmd_ttci_thwi"
        self.edmd_centers = {
            "ds": np.linspace(*self.edmd_ranges["ds"], edmd_grid_count),
            "dv": np.linspace(*self.edmd_ranges["dv"], edmd_grid_count),
            "ttci": np.linspace(*self.edmd_ranges["ttci"], edmd_grid_count),
            "thwi": np.linspace(*self.edmd_ranges["thwi"], edmd_grid_count),
        }

    def load_matrices_from_file(self, data_folder_path=None, preferred_lift_method="auto"):
        """Load A, B, C matrices from CSV files in the data_driven_workspace folder.

        Args:
            data_folder_path (str): Path to the data_driven_workspace folder.
                                   If None, uses the default relative path.
            preferred_lift_method (str): One of "auto", "sindy_baseline",
                                        "sindy_ttci_thwi", or "edmd_ttci_thwi".

        Returns:
            tuple: (A, B, C) - numpy arrays containing the loaded matrices
        """
        if data_folder_path is None:
            # Get the directory of the current script
            current_dirname = os.path.dirname(os.path.abspath(__file__))
            data_folder_path = os.path.join(current_dirname, '..', 'config', 'behavior_generation')

        # Load matrices from CSV files
        try:
            matrix_sets = []
            for filename in os.listdir(data_folder_path):
                match = re.fullmatch(r"A_(\d+)x(\d+)_matrix\.csv", filename)
                if not match:
                    continue
                rows = int(match.group(1))
                cols = int(match.group(2))
                if rows != cols:
                    continue
                b_name = f"B_{rows}x1_matrix.csv"
                c_name = f"C_1x{rows}_matrix.csv"
                a_path = os.path.join(data_folder_path, filename)
                b_path = os.path.join(data_folder_path, b_name)
                c_path = os.path.join(data_folder_path, c_name)
                if os.path.exists(b_path) and os.path.exists(c_path):
                    matrix_sets.append((rows, a_path, b_path, c_path))

            if not matrix_sets:
                raise FileNotFoundError("Could not find a supported A/B/C matrix set.")

            matrix_sets.sort(key=lambda item: item[0], reverse=True)
            selected_matrix_set = None
            if preferred_lift_method == "auto":
                selected_matrix_set = matrix_sets[0]
            elif preferred_lift_method == "sindy_baseline":
                selected_matrix_set = next((item for item in matrix_sets if item[0] == 4), None)
            elif preferred_lift_method == "sindy_ttci_thwi":
                selected_matrix_set = next((item for item in matrix_sets if item[0] == 6), None)
            elif preferred_lift_method == "edmd_ttci_thwi":
                selected_matrix_set = next((item for item in matrix_sets if item[0] not in {4, 6}), None)
            else:
                raise ValueError(
                    f"Unsupported preferred_lift_method {preferred_lift_method}. "
                    "Use auto, sindy_baseline, sindy_ttci_thwi, or edmd_ttci_thwi."
                )

            if selected_matrix_set is None:
                raise FileNotFoundError(
                    f"Could not find a matrix set matching preferred_lift_method={preferred_lift_method}."
                )

            state_dim, A_file, B_file, C_file = selected_matrix_set

            self.A = np.loadtxt(A_file, delimiter=',')
            self.B = np.loadtxt(B_file, delimiter=',')
            self.C = np.loadtxt(C_file, delimiter=',')

            # Ensure proper shape for B (should be column vector)
            if self.B.ndim == 1:
                self.B = self.B.reshape(-1, 1)

            # Ensure proper shape for C (should be row vector)
            if self.C.ndim == 1:
                self.C = self.C.reshape(1, -1)

            self._infer_lift_configuration(self.A.shape[0])

            print(f"Matrices loaded successfully from {data_folder_path}")
            print(f"A shape: {self.A.shape}, B shape: {self.B.shape}, C shape: {self.C.shape}")
            print(f"Using Koopman lift: {self.koopman_lift_method}")

            return self.A, self.B, self.C

        except FileNotFoundError as e:
            print(f"Error: Could not find matrix files in {data_folder_path}")
            print(f"Details: {e}")
            return None, None, None
        except Exception as e:
            print(f"Error loading matrices: {e}")
            return None, None, None


    def _koopman_lift_sindy(self, ds, dv, v_ego):
        ds_safe = max(float(ds), 1e-3)
        ttci = float(dv) / ds_safe
        thwi = float(v_ego) / ds_safe
        if self.koopman_lift_method == "sindy_baseline":
            return np.array([ds, dv, ds**2, dv**2], dtype=float)
        if self.koopman_lift_method == "sindy_ttci_thwi":
            return np.array([ds, dv, ds**2, dv**2, ttci, thwi], dtype=float)
        raise ValueError(f"SINDy lift requested with unsupported mode {self.koopman_lift_method}.")

    def _koopman_lift_edmd(self, ds, dv, v_ego):
        if self.edmd_centers is None:
            raise ValueError("EDMD centers are not initialized.")
        ds_safe = max(float(ds), 1e-3)
        ttci = float(dv) / ds_safe
        thwi = float(v_ego) / ds_safe
        z = []
        for c_ds in self.edmd_centers["ds"]:
            for c_dv in self.edmd_centers["dv"]:
                for c_ttci in self.edmd_centers["ttci"]:
                    for c_thwi in self.edmd_centers["thwi"]:
                        delta = np.array([ds - c_ds, dv - c_dv, ttci - c_ttci, thwi - c_thwi], dtype=float)
                        z.append(np.exp(-np.linalg.norm(delta) / (2 * self.edmd_sigma ** 2)))
        return np.array(z, dtype=float)

    def _build_koopman_lift(self, ds, dv, v_ego):
        if self.koopman_lift_method in {"sindy_baseline", "sindy_ttci_thwi"}:
            return self._koopman_lift_sindy(ds, dv, v_ego)
        if self.koopman_lift_method == "edmd_ttci_thwi":
            return self._koopman_lift_edmd(ds, dv, v_ego)
        raise ValueError(f"Unsupported Koopman lift method {self.koopman_lift_method}.")

    def perform_nonlinear_optimization_for_reward_tracking(self, Q, R, reward_target, a_max=None, a_min=None, v_max=None, v_min=None, du_max=None, du_min=None, R_du=0.0):
        """Optimize PV motion to track a reward reference using lifted state-space model.

        Q: n x n state cost weight matrix (or scalar/diagonal vector)
        R: m x m control cost weight matrix (or scalar/diagonal vector)
        reward_target: scalar target reward value or a horizon-length reward reference window
        a_max: maximum acceleration (optional)
        a_min: minimum acceleration (optional)
        v_max: maximum velocity (optional)
        v_min: minimum velocity (optional)
        du_max: maximum change in control input between steps (optional)
        du_min: minimum change in control input between steps (optional)
        R_du: cost weight for rate of change in control input (default 0.0)
        """
        # Input checks
        reward_target_array = np.asarray(reward_target, dtype=float)
        if reward_target_array.ndim == 0 or reward_target_array.size == 1:
            reward_target_window = np.full(self.h, float(reward_target_array.reshape(-1)[0]))
        else:
            reward_target_window = reward_target_array.flatten()
            if reward_target_window.shape[0] != self.h:
                raise ValueError(f"reward_target must be a scalar or a vector of length {self.h}.")

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
        lift_z = self._build_koopman_lift(ds, dv, v_ego)
        if lift_z.size != n:
            raise ValueError(
                f"Lifted state dimension {lift_z.size} does not match matrix dimension {n}. "
                "Regenerate the Koopman matrices or update the Python lifting configuration."
            )
        x0 = lift_z

        casA = casadi.DM(self.A)
        casB = casadi.DM(self.B)
        casC = casadi.DM(self.C)
        casQ = casadi.DM(Q)
        casR = casadi.DM(R)
        casRewardTarget = casadi.DM(reward_target_window)

        opti = casadi.Opti()
        x = opti.variable(n, self.h)
        u = opti.variable(m, self.h - 1)
        rw = opti.variable(self.h)
        e_r = opti.variable(self.h)
        e_acc = opti.variable(self.h - 1)

        # Initial condition
        opti.subject_to(x[:, 0] == casadi.DM(x0))
        opti.subject_to(e_r >= 0)
        opti.subject_to(e_acc >= 0)

        cost = 0
        for i in range(1, self.h):
            # State-transition
            opti.subject_to(x[:, i] == casA @ x[:, i-1] + casB @ u[:, i-1])

            # Add constraints on control input (assuming u[0] is acceleration)
            if a_max is not None:
                opti.subject_to(u[0, i-1] <= a_max + e_acc[i-1])
            if a_min is not None:
                opti.subject_to(u[0, i-1] >= a_min - e_acc[i-1])

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

            # Only the SINDy lifts retain dv explicitly as x[1].
            if self.koopman_lift_method in {"sindy_baseline", "sindy_ttci_thwi"} and n >= 2:
                if v_max is not None:
                    opti.subject_to(x[1, i] <= v_max)
                if v_min is not None:
                    opti.subject_to(x[1, i] >= v_min)

            # Reward and target tracking cost
            opti.subject_to(rw[i] == (casC @ x[:, i])[0])
            reward_target_i = casRewardTarget[i]
            cost += 1 * (rw[i] - reward_target_i)**2

            # Regularization on state and input
            cost += casadi.mtimes([x[:, i].T, casQ, x[:, i]])
            cost += casadi.mtimes([u[:, i-1].T, casR, u[:, i-1]])

            # Cost on rate of change in control input
            if i >= 2:
                du = u[:, i-1] - u[:, i-2]
                cost += R_du * casadi.mtimes([du.T, du])

            # Relaxation margin to allow tracking feasibility
            cost += 1e6 * e_r[i]**2
            cost += 1e8 * e_acc[i-1]**2
            opti.subject_to(rw[i] >= reward_target_i - e_r[i])
            opti.subject_to(rw[i] <= reward_target_i + e_r[i])

        opti.minimize(cost)
        opti.solver(self.solver, {"expand": True, "print_time": 0}, {"print_level": 0})

        sol = opti.solve()

        self.reward_tracking_x_opt = sol.value(x)
        self.reward_tracking_u_opt = sol.value(u)
        self.reward_tracking_rw_opt = sol.value(rw)
        self.reward_tracking_err_opt = sol.value(e_r)
        self.reward_tracking_acc_err_opt = sol.value(e_acc)
        self.reward_tracking_target_window = reward_target_window

        return self.reward_tracking_x_opt, self.reward_tracking_u_opt, self.reward_tracking_rw_opt

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

    def generate_braking_profile(self, jerk=-0.5, max_deceleration=-4.0):
        """Generate a smooth braking profile that ramps deceleration until the PV stops.

        Args:
            jerk (float): Constant deceleration ramp rate in m/s^3. Should be negative.
            max_deceleration (float): Lower bound on acceleration in m/s^2. Should be negative.

        Returns:
            tuple: (pv_s_opt, pv_v_opt, pv_a_opt)
        """
        if jerk >= 0:
            raise ValueError("jerk must be negative for braking.")
        if max_deceleration >= 0:
            raise ValueError("max_deceleration must be negative for braking.")

        self.pv_s_opt = np.zeros(self.h)
        self.pv_v_opt = np.zeros(self.h)
        self.pv_a_opt = np.zeros(self.h)
        self.pv_s_opt[0] = self.pv_s
        self.pv_v_opt[0] = max(self.pv_v, 0.0)
        self.pv_a_opt[0] = min(self.pv_a, 0.0)

        for i in range(1, self.h):
            if self.pv_v_opt[i - 1] <= 0.0:
                self.pv_a_opt[i] = 0.0
                self.pv_v_opt[i] = 0.0
                self.pv_s_opt[i] = self.pv_s_opt[i - 1]
                continue

            next_acc = max(self.pv_a_opt[i - 1] + jerk * self.dT, max_deceleration)
            next_v = self.pv_v_opt[i - 1] + next_acc * self.dT

            if next_v <= 0.0:
                stop_dt = self.pv_v_opt[i - 1] / max(-next_acc, 1e-6)
                stop_dt = min(stop_dt, self.dT)
                self.pv_s_opt[i] = self.pv_s_opt[i - 1] + self.pv_v_opt[i - 1] * stop_dt + 0.5 * next_acc * stop_dt ** 2
                self.pv_v_opt[i] = 0.0
                # Keep the braking command on the stopping step; later steps drop to zero once stopped.
                self.pv_a_opt[i] = next_acc
            else:
                self.pv_s_opt[i] = self.pv_s_opt[i - 1] + self.pv_v_opt[i - 1] * self.dT + 0.5 * next_acc * self.dT ** 2
                self.pv_v_opt[i] = next_v
                self.pv_a_opt[i] = next_acc

        return self.pv_s_opt, self.pv_v_opt, self.pv_a_opt

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
        opti.solver(self.solver, {"expand": True, "print_time": 0}, {"print_level": 0})

        sol = opti.solve()

        # Extract optimized values
        self.pv_a_opt = sol.value(u)
        self.pv_v_opt = sol.value(s[1, :])
        self.pv_s_opt = sol.value(s[0, :])
        self.ttci_opt = (self.pv_v_opt - self.ego_v) / (self.pv_s_opt - self.ego_s)
