#!/usr/bin/env python3

import math
import csv
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose
from dataspeed_ulc_msgs.msg import UlcCmd
from hololens_ros_communication.msg import hololens_info

def delta_yaw_correction(delta_yaw):
    if delta_yaw > math.pi:
        delta_yaw = delta_yaw - 2*math.pi
    elif delta_yaw < -math.pi:
        delta_yaw = delta_yaw + 2*math.pi
    else:
        delta_yaw = delta_yaw
    return delta_yaw


def vehicle_coordinate_transformation(goal_pose, vehicle_pose):
    dx = goal_pose[0] - vehicle_pose[0]
    dy = goal_pose[1] - vehicle_pose[1]
    v_yaw = delta_yaw_correction(goal_pose[2] - vehicle_pose[2])
    v_x = dx * math.cos(vehicle_pose[2]) + dy * math.sin(vehicle_pose[2])
    v_y = dy * math.cos(vehicle_pose[2]) - dx * math.sin(vehicle_pose[2])
    v_goal_pose = np.array([v_x, v_y, v_yaw])
    return v_goal_pose

def vehicle_coordinate_transformation_3D(goal_pose, vehicle_pose):
    # Z rotation and translation
    dx = goal_pose[0] - vehicle_pose[0]
    dy = goal_pose[1] - vehicle_pose[1]
    v_x = dx * math.cos(vehicle_pose[3]) + dy * math.sin(vehicle_pose[3])
    v_y = dy * math.cos(vehicle_pose[3]) - dx * math.sin(vehicle_pose[3])
    
    # Y rotation and translation
    dxdy = (dx**2 + dy**2)**0.5
    dz = goal_pose[2] - vehicle_pose[2]
    v_z = dz * math.cos(vehicle_pose[4]) - dxdy * math.sin(vehicle_pose[4])
    
    v_yaw = delta_yaw_correction(goal_pose[3] - vehicle_pose[3])
    v_pitch = goal_pose[4] - vehicle_pose[4]
    
    return np.array([v_x, v_y, v_z, v_yaw, v_pitch])

def global_path_reader(global_path_name):
    with open(global_path_name) as f:
        path_points = [tuple(line) for line in csv.reader(f)]
    path_points = [(float(point[0]), float(point[1]), float(
        point[2]), float(point[3])) for point in path_points]
    path_points_pos_x = [float(point[0]) for point in path_points]
    path_points_pos_y = [float(point[1]) for point in path_points]
    path_point_pos_yaw = [float(point[2]) for point in path_points]
    path_point_pos_s = [float(point[3]) for point in path_points]
    global_path = np.transpose(np.array(
        [path_points_pos_x, path_points_pos_y, path_point_pos_yaw, path_point_pos_s]))
    return global_path

def global_path_reader2(global_path_name):
    # returns [t,x,y,s,yaw,spd] map from timed map
    with open(global_path_name) as f:
        path_points = [tuple(line) for line in csv.reader(f)]
    path_points = [(float(point[0]), float(point[1]), \
                    float(point[2]), float(point[3]), float(point[4]), \
                        float(point[5]), float(point[6])) for point in path_points]
    path_points_pos_x = [float(point[1]) for point in path_points]
    path_points_pos_y = [float(point[2]) for point in path_points]
    path_point_pos_yaw = [float(point[4]) for point in path_points]
    path_point_pos_time = [float(point[0]) for point in path_points]
    path_point_pos_velocity = [float(point[5]) for point in path_points]
    
    # calculate progress along path "s"
    path_s = []
    path_s.append(0)
    for i in range(1, len(path_points_pos_x)):
        s = path_s[i-1] + np.linalg.norm([(path_points_pos_x[i]-path_points_pos_x[i-1]) , (path_points_pos_y[i]-path_points_pos_y[i-1])])
        path_s.append(s)
        
    global_path = np.transpose(np.array(
        [path_point_pos_time, path_points_pos_x, path_points_pos_y, path_s, path_point_pos_yaw, path_point_pos_velocity]))
    return global_path

def global_path_reader_mixed_reality(global_path_name):
    with open(global_path_name) as f:
        path_points = [tuple(line) for line in csv.reader(f)]
    path_points = [(float(point[0]), float(point[1]), float(
        point[2]), float(point[3]), float(point[4]), float(point[5])) for point in path_points]
    path_points_pos_x = [float(point[0]) for point in path_points]
    path_points_pos_y = [float(point[1]) for point in path_points]
    path_point_pos_z = [float(point[2]) for point in path_points]
    path_point_pos_yaw = [float(point[3]) for point in path_points]
    path_point_pos_pitch = [float(point[4]) for point in path_points]
    path_point_pos_dist = [float(point[5]) for point in path_points]
    global_path = np.transpose(np.array(
        [path_points_pos_x, path_points_pos_y, path_point_pos_z, path_point_pos_yaw, path_point_pos_pitch, path_point_pos_dist]))
    return global_path

def global_spd_profile_reader(speed_profile_name):
    with open(speed_profile_name) as f:
        path_points = [tuple(line) for line in csv.reader(f)]
    path_points = [(float(point[0]), float(point[1])) for point in path_points]
    path_points_t = [float(point[0]) for point in path_points]
    path_points_spd = [float(point[1]) for point in path_points]
    spd_profile = np.transpose(np.array([path_points_t, path_points_spd]))
    return spd_profile

def driving_cycle_spd_profile_reader(driving_cycle_profile_name):
    with open(driving_cycle_profile_name) as f:
        data_pointes = [tuple(line) for line in csv.reader(f)]
    data_pointes = [(float(point[0]), float(point[1]), float(point[2]), float(point[3])) for point in data_pointes]
    data_point_t = [float(point[0]) for point in data_pointes]
    data_point_spd = [float(point[1]) for point in data_pointes]
    data_point_acc = [float(point[2]) for point in data_pointes]
    data_point_dist = [float(point[3]) for point in data_pointes]
    spd_profile = np.transpose(np.array([data_point_t, data_point_spd, data_point_acc, data_point_dist]))
    return spd_profile

def construct_hololens_info_msg(serial_number, num_SV, Sv_id, Sv_x, Sv_y, Sv_z, Sv_pitch, Sv_yaw, Sv_vx, Sv_vy, Sv_acc, Sv_braking):
    hololens_message = hololens_info()
    hololens_message.serial = serial_number
    hololens_message.num_SVs_x = num_SV
    hololens_message.virtual_vehicle_id = Sv_id.tolist()
    hololens_message.S_v_x = Sv_x.tolist()
    hololens_message.S_v_y = Sv_y.tolist()
    hololens_message.S_v_z = Sv_z.tolist()
    hololens_message.S_v_pitch = Sv_pitch.tolist()
    hololens_message.S_v_yaw = Sv_yaw.tolist()
    hololens_message.S_v_acc = Sv_acc.tolist()
    hololens_message.S_v_brake_status = Sv_braking.tolist()
    hololens_message.S_v_vx = Sv_vx.tolist()
    hololens_message.S_v_vy = Sv_vy.tolist()
    
    return hololens_message

def goal_pose_publisher(goal_pose, N, frame_id = 'map'):
    msg = PoseArray()
    msg.header.frame_id = frame_id
    for i in range(N):
        x = goal_pose[i,0]
        y = goal_pose[i,1]
        yaw = goal_pose[i,2]
        quaternion = quaternion_from_euler(0.0, 0.0, yaw)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        msg.poses.append(pose)
    return msg

def host_vehicle_coordinate_transformation(traffic_vehicle_pose, host_vehicle_pose):
    traffic_x = traffic_vehicle_pose[0]
    traffic_y = traffic_vehicle_pose[1]
    traffic_z = traffic_vehicle_pose[2]
    traffic_yaw = traffic_vehicle_pose[3]
    traffic_pitch = traffic_vehicle_pose[4]

    host_x = host_vehicle_pose[0]
    host_y = host_vehicle_pose[1]
    host_z = host_vehicle_pose[2]
    host_yaw = host_vehicle_pose[3]
    host_pitch = host_vehicle_pose[4]

    dx = traffic_x - host_x
    dy = traffic_y - host_y
    dz = traffic_z - host_z
    dxdy = (dx**2 + dy**2)**0.5

    x_ego_coord = dx * math.cos(host_yaw) + dy * math.sin(host_yaw)
    y_ego_coord = -dx * math.sin(host_yaw) + dy * math.cos(host_yaw)
    yaw_ego_coord = yaw_change_correction(traffic_yaw - host_yaw)
    z_ego_coord = -dxdy * math.sin(host_pitch) + dz * math.cos(host_pitch)
    
    pitch_ego_coord = traffic_pitch - host_pitch

    return [x_ego_coord, y_ego_coord, z_ego_coord, yaw_ego_coord, pitch_ego_coord]

def yaw_change_correction(delta_yaw):
    if delta_yaw > math.pi:
        delta_yaw = delta_yaw - 2*math.pi
    elif delta_yaw < -math.pi:
        delta_yaw = delta_yaw + 2*math.pi
    else:
        delta_yaw = delta_yaw
    return delta_yaw

def construct_hololens_info_msg(serial_number, num_SV, Sv_id, Sv_x, Sv_y, Sv_z, Sv_pitch, Sv_yaw, Sv_vx, Sv_vy, Sv_acc, Sv_braking):
    hololens_message = hololens_info()
    hololens_message.serial = serial_number
    hololens_message.num_SVs_x = num_SV
    hololens_message.virtual_vehicle_id = Sv_id
    hololens_message.S_v_x = Sv_x
    hololens_message.S_v_y = Sv_y
    hololens_message.S_v_z = Sv_z
    hololens_message.S_v_pitch = Sv_pitch
    hololens_message.S_v_yaw = Sv_yaw
    hololens_message.S_v_acc = Sv_acc
    hololens_message.S_v_brake_status = Sv_braking
    hololens_message.S_v_vx = Sv_vx
    hololens_message.S_v_vy = Sv_vy

    return hololens_message
