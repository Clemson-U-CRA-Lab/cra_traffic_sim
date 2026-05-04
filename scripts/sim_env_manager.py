#! /usr/bin/env python3

# This is a base class to traffic simulation

# Simulation script to run CMI traffic simulation
# Subscribe to "/bridge_to_lowlevel" for ego vehicle's kinematic and dynamic motion

import rospy
from std_msgs.msg import Float64MultiArray
from hololens_ros_communication.msg import hololens_info
from cra_traffic_sim.msg import traffic_info, vehicle_traj_seq
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy

import time
import numpy as np
import math
import os
import re
import casadi
from utils import *
class stanley_vehicle_controller():
    def __init__(self, x_init, y_init, z_init, yaw_init, pitch_init, car_length):
        self.x = x_init
        self.y = y_init
        self.z = z_init
        self.yaw = yaw_init
        self.pitch = pitch_init
        self.steering = 0.0
        self.acc = 0.0
        self.v = 0.0
        self.L = car_length
    
    def update_vehicle_state(self, acc, z, pitch, dt):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v * math.tan(self.steering) / self.L * dt
        self.v += self.acc * dt
        self.z = z
        self.pitch = pitch
        self.acc = acc
    
    def pure_pursuit_controller(self, goal_pose):
        ego_pose = [self.x, self.y, self.z, self.yaw, self.pitch]
        
        local_veh_pose = host_vehicle_coordinate_transformation(goal_pose, ego_pose)
        
        l = (local_veh_pose[0] ** 2 + local_veh_pose[1] ** 2) ** 0.5
        r = l ** 2 / (2 * local_veh_pose[1])
        self.steering = np.clip(math.atan(6 / r), -0.5, 0.5)
    
    def get_traffic_pose(self):
        return [self.x, self.y, self.z, self.yaw, self.pitch]
    
class CMI_traffic_sim:
    def __init__(self, max_num_vehicles, num_vehicles, sil_simulation):
        self.serial_id = 0
        self.traffic_s = [0.0]*max_num_vehicles
        self.traffic_l = [0.0]*max_num_vehicles
        self.traffic_x = [0.0]*max_num_vehicles
        self.traffic_y = [0.0]*max_num_vehicles
        self.traffic_z = [0.0]*max_num_vehicles
        self.traffic_yaw = [0.0]*max_num_vehicles
        self.traffic_pitch = [0.0]*max_num_vehicles
        self.traffic_alon = [0.0]*max_num_vehicles
        self.traffic_v = [0.0]*max_num_vehicles
        self.traffic_omega = [0.0]*max_num_vehicles
        self.traffic_brake_status = [False]*max_num_vehicles
        self.traffic_Sv_id = [0]*max_num_vehicles
        self.traffic_num_vehicles = num_vehicles
        self.traffic_info_msg = traffic_info()
        self.vehicle_traj_msg = vehicle_traj_seq()
        self.s_init = 0.0

        self.ego_s = 0.0
        self.ego_l = 0.0
        self.ego_sv = 0.0
        self.ego_lv = 0.0
        self.ego_yaw_s = 0.0
        self.ego_v_north = 0.0
        self.ego_v_east = 0.0

        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_z = 0.0
        self.ego_pitch = 0.0
        self.ego_yaw = 0.0
        self.ego_acc = 0.0
        self.ego_omega = 0.0
        self.ego_v = 0.0
        self.ego_acceleration_pitch = 0.0
        self.ego_pose_ref = np.zeros((3, 1))

        self.traffic_initialized = False
        self.sim_start = False
        self.behavior_generation_requested = False
        self._prev_behavior_button_pressed = False
        
        if sil_simulation:
            self.sub_lowlevel_bridge = rospy.Subscriber('/odom', Odometry, self.odom_callback)
            print('Use odom as state tracker')
        else:
            self.sub_lowlevel_bridge = rospy.Subscriber('/bridge_to_lowlevel', Float64MultiArray, self.lowlevel_bridge_callback)
            print('Use lowlevel as state tracker')
            
        self.pub_traffic_info = rospy.Publisher(
            '/traffic_sim_info_mache', traffic_info, queue_size=1)
        self.sub_joy = rospy.Subscriber("/joy", Joy, self.joy_callback)
        self.pub_vehicle_traj_sequence = rospy.Publisher(
            '/front_v_traj_seq_v0', vehicle_traj_seq, queue_size=1)

    def joy_callback(self, msg):
        if msg.buttons[4]:
            self.sim_start = False
        if msg.buttons[5]:
            self.sim_start = True
        behavior_button_pressed = len(msg.buttons) > 3 and bool(msg.buttons[3])
        if behavior_button_pressed and not self._prev_behavior_button_pressed:
            self.behavior_generation_requested = True
        self._prev_behavior_button_pressed = behavior_button_pressed

    def consume_behavior_generation_request(self):
        request_active = self.behavior_generation_requested
        self.behavior_generation_requested = False
        return request_active
            
    def odom_callback(self, msg):
        self.ego_x = msg.pose.pose.position.x
        self.ego_y = msg.pose.pose.position.y
        self.ego_yaw = msg.twist.twist.angular.z
        self.ego_s = msg.pose.pose.orientation.z
        self.ego_sv = msg.twist.twist.linear.x
        self.ego_lv = msg.twist.twist.linear.y
        self.ego_v = (msg.twist.twist.linear.x ** 2 + msg.twist.twist.linear.y ** 2) ** 0.5
        self.ego_acc = msg.pose.pose.orientation.x
        self.ego_pose_ref = np.array(
            [[self.ego_x], [self.ego_y], [self.ego_z]])

    def lowlevel_bridge_callback(self, msg):
        self.ego_x = msg.data[11]
        self.ego_y = msg.data[12]
        self.ego_z = msg.data[2]
        self.ego_yaw = msg.data[5]
        self.ego_roll = msg.data[16]
        self.ego_pitch = msg.data[17]
        self.ego_v = msg.data[15]
        self.ego_acc = msg.data[6]
        self.ego_v_north = msg.data[13]
        self.ego_v_east = msg.data[14]
        self.ego_pose_ref = np.array(
            [[self.ego_x], [self.ego_y], [self.ego_z]])

    def traffic_initialization(self, s_ego, ds, line_number, vehicle_id, vehicle_id_in_lane):
        self.traffic_s[vehicle_id] = s_ego + ds * (vehicle_id_in_lane + 1)
        self.traffic_Sv_id[vehicle_id] = vehicle_id
        self.traffic_l[vehicle_id] = line_number
        self.traffic_brake_status[vehicle_id] = True
    
    def global_vehicle_update(self, veh_ID, x, y, z, yaw, pitch):
        self.traffic_x[veh_ID] = x
        self.traffic_y[veh_ID] = y
        self.traffic_z[veh_ID] = z
        self.traffic_yaw[veh_ID] = yaw
        self.traffic_pitch[veh_ID] = pitch

    def traffic_update(self, dt, a, v_tgt, vehicle_id):
        # Update velocity to match speed profile
        v_t = a * dt + self.traffic_v[vehicle_id]
        dv = v_tgt - v_t
        if (dv * a > 0):
            self.traffic_alon[vehicle_id] = a
            self.traffic_v[vehicle_id] = v_t
        else:
            self.traffic_alon[vehicle_id] = 0
            self.traffic_v[vehicle_id] = v_tgt
        # Update distance travelled using real-time velocity
        self.traffic_s[vehicle_id] = self.traffic_s[vehicle_id] + \
            self.traffic_v[vehicle_id] * dt + 0.5 * \
            self.traffic_alon[vehicle_id] * dt**2
    
    def traffic_update_from_acceleration(self, dt, a, vehicle_id):
        # Update velocity to match speed profile
        v_t = a * dt + self.traffic_v[vehicle_id]
        self.traffic_alon[vehicle_id] = a
        self.traffic_v[vehicle_id] = v_t
        # Update distance travelled using real-time velocity
        self.traffic_s[vehicle_id] = self.traffic_s[vehicle_id] + \
            self.traffic_v[vehicle_id] * dt + 0.5 * \
            self.traffic_alon[vehicle_id] * dt**2
    
    def traffic_update_from_spd_profile(self, s_t, v_t, a_t, vehicle_id):
        self.traffic_v[vehicle_id] = v_t
        self.traffic_alon[vehicle_id] = a_t
        self.traffic_s[vehicle_id] = s_t
    
    def traffic_brake_status_update(self, vehicle_id):
        # Calculate air resistance
        f_air = 0.5 * 1.225 * 0.28 * 2.6 * self.traffic_v[vehicle_id] ** 2; # air resistance
        # Calculate rolling resistance
        f_roll = 0.015 * 2250 * 9.81; # rolling resistance
        # Calculate total resistance
        f_total = f_air + f_roll
        # Calculate acceleration due to resistance
        a_resistance = -f_total / 2250
        # Calculate human brake acceleration
        a_forward_acceleration = self.traffic_alon[vehicle_id] + a_resistance
        # Update brake status
        if (a_forward_acceleration < -0.1):
            self.traffic_brake_status[vehicle_id] = True
        else:
            self.traffic_brake_status[vehicle_id] = False
    
    def ego_vehicle_frenet_update(self, s, l, sv, lv, yaw_s):
        self.ego_s = s
        self.ego_l = l
        # self.ego_sv = sv
        # self.ego_lv = lv
        self.ego_yaw_s = yaw_s
    
    def ego_acceleration_pitch_update(self, pitch_max, pitch_min, acc_max, acc_min, smoothing_factor=0.1):
        ego_acc = float(np.clip(self.ego_acc, acc_min, acc_max))
        if (ego_acc >= 0):
            acc_incited_pitch = pitch_max * (ego_acc / acc_max)
        else:
            acc_incited_pitch = pitch_min * (ego_acc / acc_min)
        acc_incited_pitch = float(np.clip(acc_incited_pitch, pitch_min, pitch_max))
        smoothing_factor = float(np.clip(smoothing_factor, 0.0, 1.0))
        self.ego_acceleration_pitch = (
            smoothing_factor * acc_incited_pitch
            + (1.0 - smoothing_factor) * self.ego_acceleration_pitch
        )
        return self.ego_acceleration_pitch
    
    def construct_vehicle_state_sequence_msg(self, id, t, s, v, a, sim_start):
        self.vehicle_traj_seq_msg = vehicle_traj_seq()
        self.vehicle_traj_seq_msg.sim_t = t
        self.vehicle_traj_seq_msg.serial = id
        
        self.vehicle_traj_seq_msg.sim_start = sim_start

        for i in range(len(v)):
            self.vehicle_traj_seq_msg.front_a[i] = a[i]
            self.vehicle_traj_seq_msg.front_v[i] = v[i]
            self.vehicle_traj_seq_msg.front_s[i] = s[i]

    def construct_traffic_sim_info_msg(self, sim_t):
        self.traffic_info_msg = traffic_info()
        self.traffic_info_msg.serial = self.serial_id
        self.traffic_info_msg.num_SVs_x = self.traffic_num_vehicles
        self.traffic_info_msg.sim_T = sim_t
        self.traffic_info_msg.virtual_vehicle_id = self.traffic_Sv_id
        self.traffic_info_msg.S_v_s = self.traffic_s
        self.traffic_info_msg.S_v_l = self.traffic_l
        self.traffic_info_msg.S_v_sv = self.traffic_v
        self.traffic_info_msg.S_v_acc = self.traffic_alon
        self.traffic_info_msg.S_v_yaw = self.traffic_yaw
        self.traffic_info_msg.S_v_omega = self.traffic_omega
        self.traffic_info_msg.S_v_brake_status = self.traffic_brake_status
        self.traffic_info_msg.E_v_s = self.ego_s
        self.traffic_info_msg.E_v_l = self.ego_l
        self.traffic_info_msg.E_v_sv = self.ego_sv
        self.traffic_info_msg.E_v_lv = self.ego_lv
        self.traffic_info_msg.E_v_yaw = self.ego_yaw_s
        self.traffic_info_msg.E_v_acc = self.ego_acc

    def publish_traffic_sim_info(self):
        self.pub_traffic_info.publish(self.traffic_info_msg)
        
    def publish_vehicle_traj(self):
        self.pub_vehicle_traj_sequence.publish(self.vehicle_traj_seq_msg)


class road_reader:
    def __init__(self, map_filename, speed_profile_filename, closed_track=False):
        self.x = []
        self.y = []
        self.z = []
        self.s = []
        self.pitch = []
        self.yaw = []

        self.t = []
        self.speed = []
        self.dist = []
        self.acc = []

        self.cmi_road_file = map_filename
        self.speed_profile_file = speed_profile_filename
        self.grand_prix_style = closed_track
        self.lane_width = 4.2
        self._map_arrays_ready = False
        self._speed_profile_arrays_ready = False
        self._last_ego_ref_index = None
        self._last_ego_frenet_index = None
        self._ego_search_window = 2000

    def read_map_data(self):
        with open(self.cmi_road_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_data = []
            for row in csv_reader:
                for element in row:
                    line_data.append(float(element))

                self.x.append(line_data[0])
                self.y.append(line_data[1])
                self.z.append(line_data[2])
                self.yaw.append(line_data[3])
                self.pitch.append(line_data[4])
                self.s.append(line_data[5])
                line_data = []
        self._refresh_map_arrays()

    def read_speed_profile(self):
        with open(self.speed_profile_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_data = []
            for row in csv_reader:
                for element in row:
                    line_data.append(float(element))

                self.t.append(line_data[0])
                self.speed.append(line_data[1])
                self.acc.append(line_data[2])
                self.dist.append(line_data[3])
                line_data = []
        self._refresh_speed_profile_arrays()

    def _refresh_map_arrays(self):
        self.x_arr = np.asarray(self.x, dtype=float)
        self.y_arr = np.asarray(self.y, dtype=float)
        self.z_arr = np.asarray(self.z, dtype=float)
        self.yaw_arr = np.asarray(self.yaw, dtype=float)
        self.pitch_arr = np.asarray(self.pitch, dtype=float)
        self.s_arr = np.asarray(self.s, dtype=float)
        self.xyz_arr = np.vstack((self.x_arr, self.y_arr, self.z_arr))
        self._map_arrays_ready = True

    def _refresh_speed_profile_arrays(self):
        self.t_arr = np.asarray(self.t, dtype=float)
        self.speed_arr = np.asarray(self.speed, dtype=float)
        self.dist_arr = np.asarray(self.dist, dtype=float)
        self.acc_arr = np.asarray(self.acc, dtype=float)
        self._speed_profile_arrays_ready = True

    def _nearest_time_index(self, query_time):
        if not self._speed_profile_arrays_ready:
            self._refresh_speed_profile_arrays()
        idx_next = int(np.searchsorted(self.t_arr, query_time, side="left"))
        if idx_next <= 0:
            return 0
        if idx_next >= len(self.t_arr):
            return len(self.t_arr) - 1
        idx_prev = idx_next - 1
        if abs(self.t_arr[idx_next] - query_time) < abs(query_time - self.t_arr[idx_prev]):
            return idx_next
        return idx_prev

    def _nearest_map_index(self, ego_poses, cache_attr):
        if not self._map_arrays_ready:
            self._refresh_map_arrays()

        ego_xyz = np.asarray(ego_poses, dtype=float).reshape(3, 1)
        num_points = len(self.s_arr)
        cached_index = getattr(self, cache_attr)
        if cached_index is None:
            start = 0
            end = num_points
        else:
            start = max(0, int(cached_index) - self._ego_search_window)
            end = min(num_points, int(cached_index) + self._ego_search_window + 1)

        local_delta = self.xyz_arr[:, start:end] - ego_xyz
        local_dist_sq = np.sum(local_delta * local_delta, axis=0)
        local_index = int(np.argmin(local_dist_sq))
        nearest_index = start + local_index

        if (
            cached_index is not None
            and ((local_index == 0 and start > 0) or (local_index == end - start - 1 and end < num_points))
        ):
            full_delta = self.xyz_arr - ego_xyz
            full_dist_sq = np.sum(full_delta * full_delta, axis=0)
            nearest_index = int(np.argmin(full_dist_sq))
            min_dist_sq = float(full_dist_sq[nearest_index])
        else:
            min_dist_sq = float(local_dist_sq[local_index])

        setattr(self, cache_attr, nearest_index)
        return nearest_index, min_dist_sq ** 0.5

    def find_traffic_vehicle_poses(self, dist_travelled, lane_id):
        if not self._map_arrays_ready:
            self._refresh_map_arrays()

        if self.grand_prix_style:
            id_virtual = int(np.argmin(np.abs(dist_travelled - self.s_arr)))
            ds = dist_travelled - self.s_arr[id_virtual]
            if (ds > 0):
                id_adjacent = (id_virtual + 1) % len(self.s_arr)
            else:
                id_adjacent = (id_virtual - 1) % len(self.s_arr)
            dist_between_map_poses = ((self.x_arr[id_adjacent] - self.x_arr[id_virtual])**2 +
                                      (self.y_arr[id_adjacent] - self.y_arr[id_virtual])**2 +
                                      (self.z_arr[id_adjacent] - self.z_arr[id_virtual])**2)**0.5
            k = np.abs(ds / dist_between_map_poses)
        else:
            id_adjacent = int(np.searchsorted(self.s_arr, dist_travelled, side="left"))
            if id_adjacent <= 0:
                id_virtual = 0
                id_adjacent = 1
            elif id_adjacent >= len(self.s_arr):
                id_virtual = len(self.s_arr) - 1
                id_adjacent = len(self.s_arr) - 2
            else:
                id_virtual = id_adjacent - 1
            k = (dist_travelled - self.s_arr[id_virtual]) / (self.s_arr[id_adjacent] - self.s_arr[id_virtual])

        traffic_x = self.x_arr[id_virtual] * (1 - k) + self.x_arr[id_adjacent] * k
        traffic_y = (self.y_arr[id_virtual] + lane_id * self.lane_width) * (1 - k) + (self.y_arr[id_adjacent] + lane_id * self.lane_width) * k
        traffic_z = self.z_arr[id_virtual] * (1 - k) + self.z_arr[id_adjacent] * k
        traffic_yaw = self.yaw_arr[id_virtual]
        traffic_pitch = self.pitch_arr[id_virtual]

        return [traffic_x, traffic_y, traffic_z, traffic_yaw, traffic_pitch]

    def find_ego_vehicle_distance_reference(self, ego_poses):
        min_ref_coordinate_id, min_dist_to_map = self._nearest_map_index(
            ego_poses,
            "_last_ego_ref_index",
        )
        s_max = np.max(self.s_arr)

        if (self.grand_prix_style):
            next_id = (min_ref_coordinate_id + 1) % len(self.s_arr)
            prev_id = (min_ref_coordinate_id - 1)
        else:
            next_id = np.clip(min_ref_coordinate_id + 1, 0, len(self.s_arr) - 1)
            prev_id = np.clip(min_ref_coordinate_id - 1, 0, len(self.s_arr) - 1)

        x_next = self.x_arr[next_id]
        y_next = self.y_arr[next_id]
        z_next = self.z_arr[next_id]

        x_prev = self.x_arr[prev_id]
        y_prev = self.y_arr[prev_id]
        z_prev = self.z_arr[prev_id]

        dist_to_next = ((ego_poses[0] - x_next)**2 + (ego_poses[1] - y_next)**2 + (ego_poses[2] - z_next)**2)**0.5
        dist_to_prev = ((ego_poses[0] - x_prev)**2 + (ego_poses[1] - y_prev)**2 + (ego_poses[2] - z_prev)**2)**0.5

        w = dist_to_next / (dist_to_next + dist_to_prev)
        s_ref = w * self.s_arr[next_id] + (1 - w) * self.s_arr[prev_id]
        return s_ref[0], min_dist_to_map, s_max

    def find_ego_frenet_pose(self, ego_poses, ego_yaw, vy, vx):
        # Find closest point from map to the ego vehicle
        min_ref_coordinate_id, _ = self._nearest_map_index(
            ego_poses,
            "_last_ego_frenet_index",
        )

        if (self.grand_prix_style):
            next_id = (min_ref_coordinate_id + 1) % len(self.s_arr)
            prev_id = (min_ref_coordinate_id - 1)
        else:
            next_id = np.clip(min_ref_coordinate_id + 1, 0, len(self.s_arr) - 1)
            prev_id = np.clip(min_ref_coordinate_id - 1, 0, len(self.s_arr) - 1)

        x_t = ego_poses[0][0]
        y_t = ego_poses[1][0]
        yaw_t = ego_yaw

        x_next = self.x_arr[next_id]
        y_next = self.y_arr[next_id]
        # z_next = self.z[next_id]
        yaw_next = self.yaw_arr[next_id]

        x_prev = self.x_arr[prev_id]
        y_prev = self.y_arr[prev_id]
        # z_prev = self.z[prev_id]
        yaw_prev = self.yaw_arr[next_id]

        d = np.abs((y_next - y_prev) * x_t - (x_next - x_prev) * y_t + x_next * y_prev - y_next * x_prev) / np.sqrt(
            (x_next - x_prev)**2 + (y_next - y_prev)**2)  # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        yaw_s = (yaw_next + yaw_prev) / 2
        E_s_yaw = yaw_t - yaw_s

        # Calculate lateral and longitudinal speed
        v_s = vx * math.cos(yaw_s) + vy * math.sin(yaw_s)
        v_l = - vx * math.sin(yaw_s) + vy * math.cos(yaw_s)
        return d, E_s_yaw, v_s, v_l

    def find_speed_profile_information(self, sim_t):
        # Locate the index of front vehicle's distance
        t_id = self._nearest_time_index(sim_t)
        dist_t = self.dist_arr[t_id]
        speed_t = self.speed_arr[t_id]
        acc_t = self.acc_arr[t_id]

        return speed_t, dist_t, acc_t
    
    def find_front_vehicle_predicted_state(self, dt, sim_t):
        # Locate the index of front vehicle's distance
        t_id = self._nearest_time_index(sim_t)
        
        # Find the time id of future states
        t_future = self.t_arr[t_id] + dt
        t_future_id = self._nearest_time_index(t_future)
        
        # Return the distance, speed and acceleration at future time step
        dist_t = self.dist_arr[t_future_id]
        speed_t = self.speed_arr[t_future_id]
        acc_t = self.acc_arr[t_future_id]

        return speed_t, dist_t, acc_t


class hololens_message_manager():
    def __init__(self, num_vehicles, max_num_vehicles, num_traffic_lights, max_num_traffic_lights):
        self.hololens_message = hololens_info()
        self.serial = 0
        self.num_SVs_x = num_vehicles
        self.num_TL = num_traffic_lights
        self.virtual_vehicle_id = [0] * max_num_vehicles
        self.S_v_x = [0.0] * max_num_vehicles
        self.S_v_y = [0.0] * max_num_vehicles
        self.S_v_z = [0.0] * max_num_vehicles
        self.S_v_pitch = [0.0] * max_num_vehicles
        self.S_v_yaw = [0.0] * max_num_vehicles
        self.S_v_acc = [0.0] * max_num_vehicles
        self.S_v_vx = [0.0] * max_num_vehicles
        self.S_v_vy = [0.0] * max_num_vehicles
        self.S_v_brake_status = [False] * max_num_vehicles

        self.TL_type = [0.0] * max_num_traffic_lights
        self.TL_ID = [0.0] * max_num_traffic_lights
        self.TL_status = [0.0] * max_num_traffic_lights
        self.TL_ds = [0.0] * max_num_traffic_lights
        self.TL_x = [0.0] * max_num_traffic_lights
        self.TL_y = [0.0] * max_num_traffic_lights
        self.TL_z = [0.0] * max_num_traffic_lights
        self.TL_pitch = [0.0] * max_num_traffic_lights
        self.TL_yaw = [0.0] * max_num_traffic_lights

        self.Ego_x = 0.0
        self.Ego_y = 0.0
        self.Ego_z = 0.0
        self.Ego_pitch = 0.0
        self.Ego_yaw = 0.0
        self.Ego_acc = 0.0
        self.Ego_omega = 0.0
        self.Ego_v = 0.0

        self.advisory_spd = 0.0

        self.pub_virtual_traffic_info = rospy.Publisher(
            '/virtual_sim_info_mache', hololens_info, queue_size=1)
    
    def update_virtual_vehicle_state(self, vehicle_id, x, y, z, pitch, yaw, acc, vx, vy, brake_status):
        self.virtual_vehicle_id[vehicle_id] = vehicle_id
        self.S_v_x[vehicle_id] = x
        self.S_v_y[vehicle_id] = y
        self.S_v_z[vehicle_id] = z
        self.S_v_pitch[vehicle_id] = pitch
        self.S_v_yaw[vehicle_id] = yaw
        self.S_v_acc[vehicle_id] = acc
        self.S_v_vx[vehicle_id] = vx
        self.S_v_vy[vehicle_id] = vy
        self.S_v_brake_status[vehicle_id] = brake_status
    
    def update_ego_state(self, serial_id, ego_x, ego_y, ego_z, ego_pitch, ego_yaw, ego_acc, ego_omega, ego_v):
        self.serial = serial_id
        self.Ego_x = ego_x
        self.Ego_y = ego_y
        self.Ego_z = ego_z
        self.Ego_pitch = ego_pitch
        self.Ego_yaw = ego_yaw
        self.Ego_acc = ego_acc
        self.Ego_omega = ego_omega
        self.Ego_v = ego_v
    
    def construct_hololens_info_msg(self):
        self.hololens_message = hololens_info()
        self.hololens_message.serial = self.serial
        self.hololens_message.num_SVs_x = self.num_SVs_x
        self.hololens_message.num_TL = self.num_TL

        for i in range(self.hololens_message.num_SVs_x):
            self.hololens_message.virtual_vehicle_id[i] = self.virtual_vehicle_id[i]
            self.hololens_message.S_v_x[i] = self.S_v_x[i]
            self.hololens_message.S_v_y[i] = self.S_v_y[i]
            self.hololens_message.S_v_z[i] = self.S_v_z[i]
            self.hololens_message.S_v_pitch[i] = self.S_v_pitch[i]
            self.hololens_message.S_v_yaw[i] = self.S_v_yaw[i]
            self.hololens_message.S_v_acc[i] = self.S_v_acc[i]
            self.hololens_message.S_v_brake_status[i] = self.S_v_brake_status[i]
            self.hololens_message.S_v_vx[i] = self.S_v_vx[i]
            self.hololens_message.S_v_vy[i] = self.S_v_vy[i]

        for j in range(self.hololens_message.num_TL):
            self.hololens_message.TL_Type[j] = self.TL_type[j]
            self.hololens_message.TL_ID[j] = self.TL_ID[j]
            self.hololens_message.TL_status[j] = self.TL_status[j]
            self.hololens_message.TL_ds[j] = self.TL_ds[j]
            self.hololens_message.TL_x[j] = self.TL_x[j]
            self.hololens_message.TL_y[j] = self.TL_y[j]
            self.hololens_message.TL_z[j] = self.TL_z[j]
            self.hololens_message.TL_pitch[j] = self.TL_pitch[j]
            self.hololens_message.TL_yaw[j] = self.TL_yaw[j]

        self.hololens_message.Ego_omega = self.Ego_omega
        self.hololens_message.Ego_acc = self.Ego_acc
        self.hololens_message.Ego_v = self.Ego_v
        self.hololens_message.Ego_x = self.Ego_x
        self.hololens_message.Ego_y = self.Ego_y
        self.hololens_message.Ego_z = self.Ego_z
        self.hololens_message.Ego_pitch = self.Ego_pitch
        self.hololens_message.Ego_yaw = self.Ego_yaw
        self.hololens_message.advisory_spd = self.advisory_spd

    def publish_virtual_sim_info(self):
        self.pub_virtual_traffic_info.publish(self.hololens_message)

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
            data_folder_path = os.path.join(current_dirname, '..', 'pv_scenario_generation_workspace', 'data_driven_workspace')
            
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
            opti.subject_to(rw[i] >= reward_target_i - e_r[i])
            opti.subject_to(rw[i] <= reward_target_i + e_r[i])

        opti.minimize(cost)
        opti.solver('fatrop', {"expand": True, "print_time": 0}, {"print_level": 0})

        sol = opti.solve()

        self.reward_tracking_x_opt = sol.value(x)
        self.reward_tracking_u_opt = sol.value(u)
        self.reward_tracking_rw_opt = sol.value(rw)
        self.reward_tracking_err_opt = sol.value(e_r)
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
        opti.solver('fatrop', {"expand": True, "print_time": 0}, {"print_level": 0})
        
        sol = opti.solve()
        
        # Extract optimized values
        self.pv_a_opt = sol.value(u)
        self.pv_v_opt = sol.value(s[1, :])
        self.pv_s_opt = sol.value(s[0, :])
        self.ttci_opt = (self.pv_v_opt - self.ego_v) / (self.pv_s_opt - self.ego_s)
