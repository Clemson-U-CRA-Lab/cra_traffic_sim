clc
clear all
dbstop if error

%%  Load the driving cycle clips
[logged_data_filename, logged_data_loc] = uigetfile('.csv', 'Please choose the rosbag.');
dc_data = readtable(strcat([logged_data_loc, logged_data_filename]));

%%  Check the acceleration data and add cruise and deceleration phaze
veh_a = dc_data.acceleration;
veh_v = dc_data.velocity;

%% Use acceleration data only for smoothness
dt = 0.1;
v_t = cumsum(veh_a * dt);
a_t = veh_a;
a = veh_a(end);

jerk_decel = -1.5;

% Reach to cruise mode
while(a >= 0)
    v_t = [v_t; v_t(end) + a*dt];
    a_t = [a_t; a];
    a = a + jerk_decel * dt;
end

% Add cruise mode
v_t = [v_t; ones(40,1) * v_t(end)];
a_t = [a_t; zeros(40,1)];
jerk_decel = -0.3;

% Add stop mode
v = v_t(end);
while(v > 0)
    v_t = [v_t; v];
    a_t = [a_t; a];
    a = a + jerk_decel * dt;
    v = v + a * dt;
end

a_t = [a_t; a_t(end)];
v_t = [v_t; 0.0];
num_data = length(v_t)-1;
sim_t = 0:0.1:0.1*num_data;

%%  Check the data
figure(1)
subplot(2,1,1)
plot(sim_t, v_t, 'LineWidth', 3);xlabel('Sim T [s]');ylabel('Speed [m/s]');grid on
subplot(2,1,2)
plot(sim_t, a_t, 'LineWidth', 3);xlabel('Sim T [s]');ylabel('Acceleration [m/s^{2}]');grid on

save_data = input('Do you want to save this clips? [0 or 1]: ');
if save_data
    data_header = {'time', 'acceleration', 'velocity'};
    data_to_save = [sim_t.', a_t, v_t];
    T = array2table(data_to_save, 'VariableNames', data_header);
    writetable(T, logged_data_filename);
end