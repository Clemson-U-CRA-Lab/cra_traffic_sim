clear
clc
close all
dbstop if error

%%  Load all driving cycles
dc_clip_struct = struct;
for i = 0:1:5
    dc_name = strcat([pwd, '/dc_clips/dc_clip_', num2str(i), '.csv']);
    dc_data = readtable(dc_name);
    dc_name = strcat(['dc_', num2str(i)]);
    dc_clip_struct.(dc_name) = dc_data;
end

%% Sort the driving cycle index randomly
dc_id = 0:1:5;
dc_id_rand = dc_id(randperm(length(dc_id)));
a = [zeros(50, 1)];
v = [zeros(50, 1)];
dt = 0.1;
for i = 1:1:6
    selected_dc_id = dc_id_rand(i);
    dc_name = strcat(['dc_', num2str(selected_dc_id)]);
    data = dc_clip_struct.(dc_name);
    wait_t = randi([3, 20], 1);
    a = [a; data.acceleration; zeros(wait_t*10, 1)];
    v = [v; data.velocity; zeros(wait_t*10, 1)];
end
sim_t = (0:dt:dt*(length(a)-1)).';
dist = [0; cumsum(v(1:end-1) + v(2:end) * dt / 2)];

%%  Sanity check the driving cycle
figure(1)
subplot(3,1,1)
plot(sim_t, v, 'LineWidth', 2);
subplot(3,1,2)
plot(sim_t, a, 'LineWidth', 2);
subplot(3,1,3)
plot(sim_t, dist, 'LineWidth', 2);

%%  Save the data
data_to_save = [sim_t, v, a, dist];
dc_profile_name = 'combined_dc_clips.csv';
writematrix(data_to_save, dc_profile_name);