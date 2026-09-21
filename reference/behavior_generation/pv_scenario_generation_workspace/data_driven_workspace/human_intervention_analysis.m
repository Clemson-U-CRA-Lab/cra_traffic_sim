clear
clc
close all
dbstop if error

%%  Load traffic data
[rosbag_name, rosbag_loc] = uigetfile('.bag', 'Please choose the rosbag.');
rosbag_loc_split = split(rosbag_loc, '\');
scenario_name = rosbag_loc_split{end-1};
disp(['Scenario name is: ', scenario_name]);
bag = rosbag(strcat([rosbag_loc, rosbag_name]));

%%  Prepare the data
traffic_info_sel = select(bag, "Topic", "/traffic_sim_info_mache");
throttle_input_sel = select(bag, "Topic", "/Mach_E/throttle_info_report");
brake_input_sel = select(bag, "Topic", "/Mach_E/brake_info_report");
mpc_parameter_sel = select(bag, "Topic", "/mpc_parameters");
control_tgt_sel = select(bag, "Topic", "/control_target_cmd");

traffic_info_struct = readMessages(traffic_info_sel, 'DataFormat', 'struct');
throttle_input_struct = readMessages(throttle_input_sel, 'DataFormat', 'struct');
brake_input_struct = readMessages(brake_input_sel, 'DataFormat', 'struct');
mpc_parameter_struct = readMessages(mpc_parameter_sel, 'DataFormat', 'struct');
control_tgt_struct = readMessages(control_tgt_sel, 'DataFormat', 'struct');

throttle_t = linspace(throttle_input_sel.StartTime, throttle_input_sel.EndTime, throttle_input_sel.NumMessages);
brake_t = linspace(brake_input_sel.StartTime, brake_input_sel.EndTime, brake_input_sel.NumMessages);

%%  Extract the data
sim_t = cellfun(@(m) double(m.SimT), traffic_info_struct);
sim_t_start = 0;
sim_t_end = 590;
[~, start_id] = min(abs(sim_t - sim_t_start));
[~, end_id] = min(abs(sim_t - sim_t_end));

front_s = cellfun(@(m) double(m.SVS(1)), traffic_info_struct);
front_v = cellfun(@(m) double(m.SVSv(1)), traffic_info_struct);

ego_s = cellfun(@(m) double(m.EVS), traffic_info_struct);
ego_vs = cellfun(@(m) double(m.EVSv), traffic_info_struct);
ego_vl = cellfun(@(m) double(m.EVLv), traffic_info_struct);
ego_v = (ego_vs .^ 2 + ego_vl .^2) .^ 0.5;
ego_a = cellfun(@(m) double(m.EVAcc), traffic_info_struct);

human_cmd = cellfun(@(m) double(m.HumanAccelerationCommand), control_tgt_struct);

ego_a = smoothdata(ego_a(start_id:end_id), 'rlowess', 20);
ego_v = ego_v(start_id:end_id);
ego_s = ego_s(start_id:end_id);
front_s = front_s(start_id:end_id);
front_v = front_v(start_id:end_id);
sim_t = sim_t(start_id:end_id);

mpc_control_cost = cellfun(@(m) double(m.ControlCost), mpc_parameter_struct);
mpc_speed_cost = cellfun(@(m) double(m.SpeedCost), mpc_parameter_struct);
mpc_sim_t = cellfun(@(m) double(m.SimT), mpc_parameter_struct);

throttle_input = cellfun(@(m) double(m.ThrottlePc), throttle_input_struct);
brake_input = cellfun(@(m) double(m.BrakeTorqueRequest), brake_input_struct);

%   Estimate power consumption
F = ego_a * 2250;
F(F<0) = 0;
P = F .* ego_v;
E = sum(P*mean(diff(sim_t))) / 1000;
disp(['Estimated power consumption is ', num2str(E), ' kJ with average power of ', ...
    num2str(mean(P)/1000), ' kW']);

%%  Check the speed
figure(1)
subplot(3,1,1)
plot(sim_t, ego_v, 'LineWidth', 2);hold on
plot(sim_t, front_v, '-.r', 'LineWidth', 2.5);
xlabel('Time [s]');ylabel('Speed [m/s]')

subplot(3,1,2)
plot(sim_t, front_s - ego_s, '-', 'LineWidth', 2, 'DisplayName', sprintf(strcat(['Average distance headway: ', num2str(mean(front_s - ego_s))])));hold on
xlabel('Time [s]');ylabel('Distance headway [m]')
legend show

subplot(3,1,3)
plot(sim_t, ego_a, '-', 'LineWidth', 2);hold on
xlabel('Time [s]');ylabel('Ego Acceleration [m/{s^2}]')

%%  Check the speed gap and distance headway during intervention
% Find human control input
control_tgt_sim_t = cellfun(@(m) double(m.SimT), control_tgt_struct);
human_acc_cmd = cellfun(@(m) double(m.HumanAccelerationCommand), control_tgt_struct);

% Find speed gap
sim_t = cellfun(@(m) double(m.SimT), traffic_info_struct);
front_v = cellfun(@(m) double(m.SVSv(1)), traffic_info_struct);
ego_vs = cellfun(@(m) double(m.EVSv), traffic_info_struct);
ego_vl = cellfun(@(m) double(m.EVLv), traffic_info_struct);
ego_v = (ego_vs .^ 2 + ego_vl .^2) .^ 0.5;

% Find distance headway
front_s = cellfun(@(m) double(m.SVS(1)), traffic_info_struct);
ego_s = cellfun(@(m) double(m.EVS), traffic_info_struct);

% Find acceleration
front_a = cellfun(@(m) double(m.SVAcc(1)), traffic_info_struct);
ego_a = smoothdata(cellfun(@(m) double(m.EVAcc), traffic_info_struct), 'gaussian', 25);

% Find time to collision
ttci = (ego_v - front_v) ./ (front_s - ego_s);

ds = front_s - ego_s;
dv = front_v - ego_v;

figure(3)
yyaxis left
plot(sim_t, ttci);hold on
yyaxis right
plot(sim_t, ego_a);

figure(4)
subplot(2,1,1)
plot(brake_t - brake_t(1), brake_input/5000, 'LineWidth', 2);grid on
xlabel('Time [s]');ylabel('Brake Percentage [%]')
subplot(2,1,2)
plot(throttle_t - throttle_t(1), throttle_input, 'LineWidth', 2);grid on
xlabel('Time [s]');ylabel('Acceleration Percentage [%]')