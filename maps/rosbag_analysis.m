%%  This script is designed to read the message from recorded rosbag file
clear
clc
close all
dbstop if error

linewidth = 3;
textsize = 20;
loop_map = false;
mapping = true;
save_mapping = false;

%%  Load the data
% Load rosbag file
bag_filename = 'itic_dir0_lane_change.bag';

% Read the rosbag data
bag = rosbag(bag_filename);

low_level_msg = '/bridge_to_lowlevel';

%% Get novatel topics
bestpos = select(bag,'Topic','/novatel/oem7/bestpos');
bestpos_struct = readMessages(bestpos,'DataFormat','struct');

corrimu = select(bag,'Topic','/novatel/oem7/corrimu');
corrimu_struct = readMessages(corrimu,'DataFormat','struct');

odom = select(bag,'Topic','/novatel/oem7/odom');
odom_struct = readMessages(odom,'DataFormat','struct');

time = select(bag,'Time',...
    [bag.StartTime bag.EndTime],'Topic','/novatel/oem7/time');
time_struct = readMessages(time,'DataFormat','struct');

imu = select(bag, 'Topic', '/gps/imu');
imu_struct = readMessages(imu, 'DataFormat', 'struct');

%%  Convert to the data structure
low_level_data = select(bag, 'Topic', low_level_msg);
low_level_struct = readMessages(low_level_data, 'DataFormat', 'struct');

%%  Data analysis
ego_x = cellfun(@(m) double(m.Data(1)), low_level_struct);
ego_y = cellfun(@(m) double(m.Data(2)), low_level_struct);
ego_z = cellfun(@(m) double(m.Data(3)), low_level_struct);
ego_x_pred = cellfun(@(m) double(m.Data(12)), low_level_struct);
ego_y_pred = cellfun(@(m) double(m.Data(13)), low_level_struct);

ego_v_hor = cellfun(@(m) double(m.Data(4)), low_level_struct);
ego_v_ver = cellfun(@(m) double(m.Data(5)), low_level_struct);
ego_merge_spd = cellfun(@(m) double(m.Data(16)), low_level_struct);
ego_spd_pred = cellfun(@(m) double(m.Data(10)), low_level_struct);


ego_yaw = cellfun(@(m) double(m.Data(6)), low_level_struct);
ego_a_lon = cellfun(@(m) double(m.Data(7)), low_level_struct);
ego_a_lat = cellfun(@(m) double(m.Data(8)), low_level_struct);
ego_pitch = cellfun(@(m) double(m.Data(18)), low_level_struct);

ego_id = 1:1:length(ego_x);

%% Generate maps
if mapping
    s_map = [0];
    x_map = [ego_x(1)];
    y_map = [ego_y(1)];
    z_map = [ego_z(1)];
    yaw_map = [ego_yaw(1)];
    pitch_map = [ego_pitch(1)];
    

    for i = 2:1:length(ego_x)
        ds = ((ego_x(i) - x_map(end))^2 + (ego_y(i) - y_map(end))^2)^0.5;
        if (ds > 0.2)
            x_map = [x_map, ego_x(i)];
            y_map = [y_map, ego_y(i)];
            z_map = [z_map, ego_z(i)];
            yaw_map = [yaw_map, ego_yaw(i)];
            pitch_map = [pitch_map, ego_pitch(i)];
            s_map = [s_map, s_map(end) + ds];
        else
            continue
        end
    end

    pitch_map = smoothdata(pitch_map, 'gaussian', 10);

    figure(1)
    scatter3(x_map, y_map, z_map, 'filled');
    xlabel('X [m]');ylabel('Y [m]');
    
    figure(2)
    subplot(6,1,1)
    plot(x_map)
    ylabel('X [m]')
    subplot(6,1,2)
    plot(y_map)
    ylabel('Y [m]')
    subplot(6,1,3)
    plot(z_map)
    ylabel('Z [m]')
    subplot(6,1,4)
    plot(s_map)
    ylabel('Distance [m]')
    subplot(6,1,5)
    plot(pitch_map)
    ylabel('Pitch [rad]')
    subplot(6,1,6)
    plot(yaw_map)
    ylabel('Yaw [rad]')

    % Save the data
    if save_mapping
        data_to_save = [x_map.', y_map.', z_map.', yaw_map.', pitch_map.', s_map.'];
        writematrix(data_to_save, map_file_name);
    end
end

%%  Trajectory comparison
figure(3)
scatter(ego_id, ego_v_hor);hold on
scatter(ego_id, ego_merge_spd);
legend('Ego v hor', 'Ego merge spd')
% scatter3(ego_x, ego_y, ego_z, 20, "red", "filled")
% legend('Map Reference', 'Recoreded Trajectory', 'FontSize', 20)