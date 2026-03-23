clear
clc
close all
dbstop if error

%%  Load traffic data
[logged_data_filename, logged_data_loc] = uigetfile('.csv', 'Please choose the rosbag.');
logged_loc_split = split(logged_data_loc, '\');
scenario_name = logged_loc_split{end-1};
disp(['Scenario name is: ', scenario_name]);
traffic_data = load(strcat([logged_data_loc, logged_data_filename]));

%%  Extract the data
sim_t = 0:0.1:0.1*(length(traffic_data(:, 1))-1);

front_s = traffic_data(:, 7);
front_v = traffic_data(:, 6);
front_a = traffic_data(:, 5);

ego_s = traffic_data(:, 4);
ego_v = traffic_data(:, 3);
ego_a = traffic_data(:, 2);

human_cmd_acc = traffic_data(:, 8);
human_cmd_brake = traffic_data(:, 9);

%% Assign reward to front vehicle's motion
ds = front_s - ego_s;
dv = front_v - ego_v;
da = front_a - ego_a;

ttci = dv ./ ds;
thwi = ego_v ./ ds;

% Find the moment of human intervention
t_intervene_pull = (human_cmd_acc ~= 0);
t_intervene_pull_diff = diff(t_intervene_pull);
t_intervene_pull_start_id = find(t_intervene_pull_diff == 1);

% Assign acceleration mismatch reward
da_reward = smoothdata(1*abs(da), 'rlowess', 10);

% Assign human intervention reward
intervention_reward = zeros(length(da_reward), 1);

state_before_intervention = struct;

for i = 1:1:length(t_intervene_pull_start_id)
    start_id_2 = max(t_intervene_pull_start_id(i) - 20, 1);
    start_id_1 = max(t_intervene_pull_start_id(i) - 10, 1);
    start_id = t_intervene_pull_start_id(i);
    intervention_reward(start_id_2:start_id_1) = 0:1:1*(start_id_1 - start_id_2);

    if start_id_1 > 1
        intervention_reward(start_id_1:start_id) = intervention_reward(start_id_1 - 1) + (0:1:1*(start_id - start_id_1));
    else
        intervention_reward(start_id_1:start_id) = 0:1:1*(start_id - start_id_1);
    end

    field_name = strcat(['event_', num2str(i)]);
    state_before_intervention.(field_name).time = sim_t(start_id_2:start_id) - sim_t(start_id);
    state_before_intervention.(field_name).ds = ds(start_id_2:start_id);
    state_before_intervention.(field_name).dv = dv(start_id_2:start_id);
    state_before_intervention.(field_name).da = da(start_id_2:start_id);
    state_before_intervention.(field_name).ttci = ttci(start_id_2:start_id);
    state_before_intervention.(field_name).thwi = thwi(start_id_2:start_id);
    state_before_intervention.(field_name).intervention_start_time = sim_t(start_id);
    state_before_intervention.(field_name).intervention_start_id = start_id;
end

reward = da_reward + intervention_reward;

figure(1)
subplot(2,1,1)
plot(sim_t, dv);
subplot(2,1,2)
plot(sim_t, ttci);

figure(2)
yyaxis left
plot(sim_t, reward, '-r', 'LineWidth', 2);hold on
ylabel('Human Intervention Reward')
yyaxis right
plot(sim_t, human_cmd_acc, '-k', 'LineWidth', 2);
ylabel('Human Acceleration [m/s^{2}]')
ax = gca;
ax.YAxis(1).Color = 'r';
ax.YAxis(2).Color = 'k';
xlabel('Time [s]');xlim([0, 220])

%%  Check thw changes and ttc changes
figure(3)
event_fields = fieldnames(state_before_intervention);

subplot(3,2,1)
hold on
for i = 1:1:length(event_fields)
    event_data = state_before_intervention.(event_fields{i});
    plot(event_data.time, event_data.ttci, 'LineWidth', 2, ...
        'DisplayName', strrep(event_fields{i}, '_', ' '));
end

grid on
xlabel('Time to intervention [s]')
ylabel('Time-to-collision inverse [s^{-1}]')
title('TTCI During 5 Seconds Before Human Intervention')
legend show

subplot(3,2,2)
hold on
for i = 1:1:length(event_fields)
    event_data = state_before_intervention.(event_fields{i});
    plot(event_data.time, event_data.thwi, 'LineWidth', 2, ...
        'DisplayName', strrep(event_fields{i}, '_', ' '));
end

grid on
xlabel('Time to intervention [s]')
ylabel('Time-headway inverse [s^{-1}]')
title('THWI During 5 Seconds Before Human Intervention')
legend show

subplot(3,2,3)
hold on
for i = 1:1:length(event_fields)
    event_data = state_before_intervention.(event_fields{i});
    plot(event_data.time, event_data.dv, 'LineWidth', 2, ...
        'DisplayName', strrep(event_fields{i}, '_', ' '));
end

grid on
xlabel('Time to intervention [s]')
ylabel('Speed gap [m/s]')
title('Speed Gap During 5 Seconds Before Human Intervention')
legend show

subplot(3,2,4)
hold on
for i = 1:1:length(event_fields)
    event_data = state_before_intervention.(event_fields{i});
    plot(event_data.time, event_data.ds, 'LineWidth', 2, ...
        'DisplayName', strrep(event_fields{i}, '_', ' '));
end

grid on
xlabel('Time to intervention [s]')
ylabel('Distance headway [m]')
title('Distance Headway During 5 Seconds Before Human Intervention')
legend show

subplot(3,2,5)
hold on
for i = 1:1:length(event_fields)
    event_data = state_before_intervention.(event_fields{i});
    plot(event_data.time, event_data.da, 'LineWidth', 2, ...
        'DisplayName', strrep(event_fields{i}, '_', ' '));
end

grid on
xlabel('Time to intervention [s]')
ylabel('Acceleration gap [m/s^2]')
title('Acceleration Gap During 5 Seconds Before Human Intervention')
legend show

%%  Save the state to approximate the dynamics on lifted space.
data_header = {'space_error', 'speed_error', 'ego_speed', 'ego_acceleration', 'front_speed', 'front_acceleration', 'reward'};
data_to_save = [ds, dv, ego_v, ego_a, front_v, front_a, reward];
filename = strcat([scenario_name, '_reward_trajectory.csv']);
T = array2table(data_to_save, 'VariableNames', data_header);
writetable(T, filename);
