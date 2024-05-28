#! /usr/bin/env python3

# This is a base class to traffic simulation

# Simulation script to run CMI traffic simulation
# Subscribe to "/bridge_to_lowlevel" for ego vehicle's kinematic and dynamic motion

import rospy
from std_msgs.msg import Float64MultiArray
from hololens_ros_communication.msg import hololens_info

import time
import numpy as np
import math
import os
from utils import *

class CMI_traffic_sim:
    def __init__(self, num_vehicles):
        self.serial_id = 0
        self.traffic_s = np.zeros((1, num_vehicles)).tolist()[0]
        self.traffic_l = np.zeros((1, num_vehicles)).tolist()[0]
        self.traffic_yaw = np.zeros((1, num_vehicles)).tolist()[0]
        self.traffic_pitch = np.zeros((1, num_vehicles)).tolist()[0]
        self.traffic_alon = np.zeros((1, num_vehicles)).tolist()[0]
        self.traffic_v = np.zeros((1, num_vehicles)).tolist()[0]
        self.traffic_omega = np.zeros((1, num_vehicles)).tolist()[0]
        self.traffic_brake_status = np.zeros((1, num_vehicles)).tolist()[0]
        self.traffic_num_vehicles = num_vehicles
        self.traffic_Sv_id = np.zeros((1, num_vehicles)).tolist()[0]

        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_z = 0.0
        self.ego_pitch = 0.0
        self.ego_yaw = 0.0
        self.ego_acc = 0.0
        self.ego_omega = 0.0
        self.ego_v = 0.0

        self.traffic_initialized = False

        self.sub_lowlevel_bridge = rospy.Subscriber(
            '/bridge_to_lowlevel', Float64MultiArray, self.lowlevel_bridge_callback)

    def lowlevel_bridge_callback(self, msg):
        self.ego_x = msg.data[11]
        self.ego_y = msg.data[12]
        self.ego_z = msg.data[2]
        self.ego_yaw = msg.data[5]
        self.ego_roll = msg.data[16]
        self.ego_pitch = msg.data[17]
        self.ego_v_lon = msg.data[15]

    def traffic_initialization(self, s_ego, ds, line_number, vehicle_id, vehicle_id_in_lane):
        self.traffic_s[vehicle_id] = s_ego + ds * (vehicle_id_in_lane + 1)
        self.traffic_Sv_id[vehicle_id] = vehicle_id
        self.traffic_l[vehicle_id] = line_number

    def traffic_update(self, dt, a, v_tgt, vehicle_id, ds):
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
        self.traffic_s[vehicle_id] = self.traffic_s[vehicle_id] + self.traffic_v[vehicle_id] * dt + 0.5 * self.traffic_alon[vehicle_id] * dt**2

class cmi_road_reader:
    def __init__(self, map_filename, speed_profile_filename, closed_track = False):
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

    def find_traffic_vehicle_poses(self, dist_travelled):
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

        k = abs(ds / dist_between_map_poses)

        traffic_x = self.x[id_virtual] * (1 - k) + self.x[id_adjacent] * k
        traffic_y = self.y[id_virtual] * (1 - k) + self.y[id_adjacent] * k
        traffic_z = self.z[id_virtual] * (1 - k) + self.z[id_adjacent] * k
        traffic_yaw = self.yaw[id_virtual]
        traffic_pitch = self.pitch[id_virtual]

        return [traffic_x, traffic_y, traffic_z, traffic_yaw, traffic_pitch]

    def find_ego_vehicle_distance_reference(self, ego_poses):

        cmi_traj_coordinate = np.array([self.x, self.y, self.z])
        dist_to_map = np.linalg.norm(cmi_traj_coordinate - ego_poses, axis=0)
        min_ref_coordinate_id = np.argmin(dist_to_map)
        min_dist_to_map = np.min(dist_to_map)

        if (self.grand_prix_style):
            next_id = (min_ref_coordinate_id + 1) % len(self.s)
            prev_id = (min_ref_coordinate_id - 1)
        else:
            next_id = np.clip(min_ref_coordinate_id + 1, 0, len(self.s))
            prev_id = np.clip(min_ref_coordinate_id - 1, 0, len(self.s))

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

        return s_ref, min_dist_to_map

    def find_speed_profile_information(self, sim_t):
        record_t = np.array(self.t)
        t_id = np.argmin(np.abs(record_t - sim_t))
        dist_t = self.dist[t_id]
        speed_t = self.speed[t_id]
        acc_t = self.acc[t_id]

        return speed_t, dist_t, acc_t

class hololens_message_manager():
    def __init__(self, num_vehicles, max_num_vehicles):
        self.hololens_message = hololens_info()
        self.serial = 0
        self.num_SVs_x = num_vehicles
        self.virtual_vehicle_id = np.zeros((1, max_num_vehicles), dtype=int).tolist()[0]
        self.S_v_x = np.zeros((1, max_num_vehicles), dtype=float).tolist()[0]
        self.S_v_y = np.zeros((1, max_num_vehicles), dtype=float).tolist()[0]
        self.S_v_z = np.zeros((1, max_num_vehicles), dtype=float).tolist()[0]
        self.S_v_pitch = np.zeros((1, max_num_vehicles), dtype=float).tolist()[0]
        self.S_v_yaw = np.zeros((1, max_num_vehicles), dtype=float).tolist()[0]
        self.S_v_acc = np.zeros((1, max_num_vehicles), dtype=float).tolist()[0]
        self.S_v_vx = np.zeros((1, max_num_vehicles), dtype=float).tolist()[0]
        self.S_v_vy = np.zeros((1, max_num_vehicles), dtype=float).tolist()[0]
        self.S_v_brake_status = np.zeros((1, max_num_vehicles), dtype=bool).tolist()[0]

        self.Ego_x = 0.0
        self.Ego_y = 0.0
        self.Ego_z = 0.0
        self.Ego_pitch = 0.0
        self.Ego_yaw = 0.0
        self.Ego_acc = 0.0
        self.Ego_omega = 0.0
        self.Ego_v = 0.0

        self.pub_virtual_traffic_info = rospy.Publisher(
            '/virtual_sim_info', hololens_info, queue_size=1)
        
    def construct_hololens_info_msg(self):
        self.hololens_message.serial = self.serial
        self.hololens_message.num_SVs_x = self.num_SVs_x
        self.hololens_message.virtual_vehicle_id = self.virtual_vehicle_id
        self.hololens_message.S_v_x = self.S_v_x
        self.hololens_message.S_v_y = self.S_v_y
        self.hololens_message.S_v_z = self.S_v_z
        self.hololens_message.S_v_pitch = self.S_v_pitch
        self.hololens_message.S_v_yaw = self.S_v_yaw
        self.hololens_message.S_v_acc = self.S_v_acc
        self.hololens_message.S_v_vx = self.S_v_vx
        self.hololens_message.S_v_vy = self.S_v_vy
        self.hololens_message.S_v_brake_status = self.S_v_brake_status

        self.hololens_message.Ego_acc = self.Ego_acc
        self.hololens_message.Ego_omega = self.Ego_omega
        self.hololens_message.Ego_v = self.Ego_v
        self.hololens_message.Ego_x = self.Ego_x
        self.hololens_message.Ego_y = self.Ego_y
        self.hololens_message.Ego_z = self.Ego_z
        self.hololens_message.Ego_pitch = self.Ego_pitch
        self.hololens_message.Ego_yaw = self.Ego_yaw
    
    def publish_virtual_sim_info(self):
        self.pub_virtual_traffic_info.publish(self.hololens_message)