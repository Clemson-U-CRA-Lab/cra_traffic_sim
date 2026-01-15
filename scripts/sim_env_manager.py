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
        self.traffic_s = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.traffic_l = np.zeros(max_num_vehicles, dtype=float).tolist()
        
        self.traffic_x = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.traffic_y = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.traffic_z = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.traffic_yaw = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.traffic_pitch = np.zeros(max_num_vehicles, dtype=float).tolist()
        
        self.traffic_alon = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.traffic_v = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.traffic_omega = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.traffic_brake_status = np.zeros(max_num_vehicles, dtype=bool).tolist()
        self.traffic_num_vehicles = num_vehicles
        self.traffic_Sv_id = np.zeros(max_num_vehicles, dtype=int).tolist()
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
        self.ego_pose_ref = np.zeros((3, 1))

        self.traffic_initialized = False
        self.sim_start = False
        
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
    
    def ego_acceleration_pitch_update(self, pitch_max, pitch_min, acc_max, acc_min):
        if (self.ego_acc >= 0):
            acc_incited_pitch = pitch_max * (self.ego_acc / acc_max)
        else:
            acc_incited_pitch = pitch_min * (self.ego_acc / acc_min)
        return acc_incited_pitch
    
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

    def find_traffic_vehicle_poses(self, dist_travelled, lane_id):
        dist_map = np.array(self.s)
        id_virtual = np.argmin(np.abs(dist_travelled - dist_map))
        ds = dist_travelled - self.s[id_virtual]

        if (ds > 0):
            id_adjacent = (id_virtual + 1) % len(self.s)
        else:
            id_adjacent = (id_virtual - 1) % len(self.s)

        dist_between_map_poses = ((self.x[id_adjacent] - self.x[id_virtual])**2 +
                                  (self.y[id_adjacent] - self.y[id_virtual])**2 +
                                  (self.z[id_adjacent] - self.z[id_virtual])**2)**0.5

        k = np.abs(ds / dist_between_map_poses)

        traffic_x = self.x[id_virtual] * (1 - k) + self.x[id_adjacent] * k
        traffic_y = (self.y[id_virtual] + lane_id * 3.8) * (1 - k) + (self.y[id_virtual] + lane_id * 3.8) * k
        traffic_z = self.z[id_virtual] * (1 - k) + self.z[id_adjacent] * k
        traffic_yaw = self.yaw[id_virtual]
        traffic_pitch = self.pitch[id_virtual]

        return [traffic_x, traffic_y, traffic_z, traffic_yaw, traffic_pitch]

    def find_ego_vehicle_distance_reference(self, ego_poses):
        traj_coordinate = np.array([self.x, self.y, self.z])
        dist_to_map = np.linalg.norm(traj_coordinate - ego_poses, axis=0)
        min_ref_coordinate_id = np.argmin(dist_to_map)
        min_dist_to_map = np.min(dist_to_map)
        s_max = np.max(self.s)

        if (self.grand_prix_style):
            next_id = (min_ref_coordinate_id + 1) % len(self.s)
            prev_id = (min_ref_coordinate_id - 1)
        else:
            next_id = np.clip(min_ref_coordinate_id + 1, 0, len(self.s) - 1)
            prev_id = np.clip(min_ref_coordinate_id - 1, 0, len(self.s) - 1)

        x_next = self.x[next_id]
        y_next = self.y[next_id]
        z_next = self.z[next_id]

        x_prev = self.x[prev_id]
        y_prev = self.y[prev_id]
        z_prev = self.z[prev_id]

        dist_to_next = ((ego_poses[0] - x_next)**2 + (ego_poses[1] - y_next)**2 + (ego_poses[2] - z_next)**2)**0.5
        dist_to_prev = ((ego_poses[0] - x_prev)**2 + (ego_poses[1] - y_prev)**2 + (ego_poses[2] - z_prev)**2)**0.5

        w = dist_to_next / (dist_to_next + dist_to_prev)
        s_ref = w * self.s[next_id] + (1 - w) * self.s[prev_id]
        return s_ref[0], min_dist_to_map, s_max

    def find_ego_frenet_pose(self, ego_poses, ego_yaw, vy, vx):
        # Find closest point from map to the ego vehicle
        cmi_traj_coordinate = np.array([self.x, self.y, self.z])
        dist_to_map = np.linalg.norm(cmi_traj_coordinate - ego_poses, axis=0)
        min_ref_coordinate_id = np.argmin(dist_to_map)

        if (self.grand_prix_style):
            next_id = (min_ref_coordinate_id + 1) % len(self.s)
            prev_id = (min_ref_coordinate_id - 1)
        else:
            next_id = np.clip(min_ref_coordinate_id + 1, 0, len(self.s) - 1)
            prev_id = np.clip(min_ref_coordinate_id - 1, 0, len(self.s) - 1)

        x_t = ego_poses[0][0]
        y_t = ego_poses[1][0]
        yaw_t = ego_yaw

        x_next = self.x[next_id]
        y_next = self.y[next_id]
        # z_next = self.z[next_id]
        yaw_next = self.yaw[next_id]

        x_prev = self.x[prev_id]
        y_prev = self.y[prev_id]
        # z_prev = self.z[prev_id]
        yaw_prev = self.yaw[next_id]

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
        record_t = np.array(self.t)
        t_id = np.argmin(np.abs(record_t - sim_t))
        dist_t = self.dist[t_id]
        speed_t = self.speed[t_id]
        acc_t = self.acc[t_id]

        return speed_t, dist_t, acc_t
    
    def find_front_vehicle_predicted_state(self, dt, sim_t):
        # Locate the index of front vehicle's distance
        front_t = np.array(self.t)
        t_id = np.argmin(np.abs(front_t - sim_t))
        
        # Find the time id of future states
        t_ref = np.array(self.t)
        t_future = self.t[t_id] + dt
        t_future_id = np.argmin(np.abs(t_ref - t_future))
        
        # Return the distance, speed and acceleration at future time step
        dist_t = self.dist[t_future_id]
        speed_t = self.speed[t_future_id]
        acc_t = self.acc[t_future_id]

        return speed_t, dist_t, acc_t


class hololens_message_manager():
    def __init__(self, num_vehicles, max_num_vehicles, num_traffic_lights, max_num_traffic_lights):
        self.hololens_message = hololens_info()
        self.serial = 0
        self.num_SVs_x = num_vehicles
        self.num_TL = num_traffic_lights
        self.virtual_vehicle_id = np.zeros(max_num_vehicles, dtype=int).tolist()
        self.S_v_x = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.S_v_y = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.S_v_z = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.S_v_pitch = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.S_v_yaw = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.S_v_acc = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.S_v_vx = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.S_v_vy = np.zeros(max_num_vehicles, dtype=float).tolist()
        self.S_v_brake_status = np.zeros(max_num_vehicles, dtype=bool).tolist()

        self.TL_type = np.zeros(max_num_traffic_lights, dtype=float).tolist()
        self.TL_ID = np.zeros(max_num_traffic_lights, dtype=float).tolist()
        self.TL_status = np.zeros(max_num_traffic_lights, dtype=float).tolist()
        self.TL_ds = np.zeros(max_num_traffic_lights, dtype=float).tolist()

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

        self.hololens_message.Ego_v = self.Ego_v
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