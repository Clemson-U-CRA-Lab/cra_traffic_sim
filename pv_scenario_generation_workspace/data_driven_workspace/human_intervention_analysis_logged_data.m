
%%  Load traffic data
[logged_data_filename, logged_data_loc] = uigetfile('.csv', 'Please choose the rosbag.');
logged_loc_split = split(logged_data_loc, '\');
scenario_name = logged_loc_split{end-1};
disp(['Scenario name is: ', scenario_name]);
traffic_data = load(strcat([logged_data_loc, logged_data_filename]));

%%  Load speed profile data
scenario_name_split = split(scenario_name, '_');
dc_name = scenario_name_split{1};
full_dc_name = strcat(['HAMPC_custom_', dc_name, '.csv']);
dc_data = load(full_dc_name);
dc_t = dc_data(:, 1);
dc_spd = dc_data(:, 2);
dc_acc = dc_data(:, 3);
dc_dist = dc_data(:, 4);

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

% Estimate power consumption
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
% plot(dc_t, dc_spd, '-.r', 'LineWidth', 2.5);
xlabel('Time [s]');ylabel('Speed [m/s]')

subplot(3,1,2)
plot(sim_t, front_s - ego_s, '-', 'LineWidth', 2, 'DisplayName', sprintf(strcat(['Average distance headway: ',num2str(mean(front_s - ego_s))])));hold on
xlabel('Time [s]');ylabel('Distance headway [m]')
legend show

subplot(3,1,3)
plot(sim_t, ego_a, '-', 'LineWidth', 2);hold on
xlabel('Time [s]');ylabel('Ego Acceleration [m/{s^2}]')

figure(2)
subplot(2,1,1)
plot(sim_t, human_cmd_acc, '-k', 'LineWidth', 2);
xlabel('Time [s]');ylabel('Accelerator pedal percentage [%]')

subplot(2,1,2)
plot(sim_t, human_cmd_brake, '-k', 'LineWidth', 2);
xlabel('Time [s]');ylabel('Brake torque request [Nm]')

%%  Check the speed gap and distance headway during intervention
% Find time to collision
ttci = (ego_v - front_v) ./ (front_s - ego_s);

ds = front_s - ego_s;
dv = front_v - ego_v;

figure(3)
yyaxis left
plot(sim_t, ttci);hold on
yyaxis right
plot(sim_t, human_cmd_acc);

%%  Check the interaction between front vehicle and rear vehicle
% Find the period when human intervene
human_intervene_t_id = (human_cmd_acc ~= 0) + (human_cmd_brake ~= 0);
human_intervene_t_id_diff = diff(human_intervene_t_id);
human_intervene_start_id = find(human_intervene_t_id_diff == 1);
humna_intervene_end_id = find(human_intervene_t_id_diff == -1);

human_intervene_data = struct;

for i = 1:1:length(human_intervene_start_id)
    field = strcat(['sec' , num2str(i)]);
    value = [front_s(human_intervene_start_id(i)-20: humna_intervene_end_id(i)), ...
             front_v(human_intervene_start_id(i)-20: humna_intervene_end_id(i)), ...
             front_a(human_intervene_start_id(i)-20: humna_intervene_end_id(i)), ...
             ego_s(human_intervene_start_id(i)-20: humna_intervene_end_id(i)), ...
             ego_v(human_intervene_start_id(i)-20: humna_intervene_end_id(i)), ...
             ego_a(human_intervene_start_id(i)-20: humna_intervene_end_id(i))];
    human_intervene_data.(field) = value;
end

%%  Check the human operation during intervention
ttci_init = [];
ttci_end = [];

for i = 1:1:length(human_intervene_start_id)
    fieldname = strcat(['sec' , num2str(i)]);
    value = human_intervene_data.(fieldname);
    
    pv_s = value(:, 1);
    pv_v = value(:, 2);
    pv_a = value(:, 3);

    ego_s = value(:, 4);
    ego_v = value(:, 5);
    ego_a = value(:, 6);

    ttci = smoothdata((pv_v - ego_v) ./ (pv_s - ego_s), 'rlowess', 20);
    thwi = smoothdata(ego_v ./ (pv_s - ego_s), 'rlowess', 20);
    ttci_dot = [0; diff(ttci) / 0.1];
    ds = pv_s - ego_s;
    dv = pv_v - ego_v;
    da = pv_a - ego_a;
    t = 0:0.1:0.1*(length(pv_s) - 1);

    ttci_init = [ttci_init; ttci(1)];
    ttci_end = [ttci_end; ttci(end)];
    
    figure(4)
    scatter(ds(14:15), ttci(14:15), 30, dv(14:15), 'filled');hold on
    xlabel('Distance headway [m]')
    ylabel('Time-to-collision Inverse [s^{-1}]')
    grid on
    c = colorbar;
    c.Label.String = 'Speed gap [m/s]';
    ds_record = [ds_record; ds(14:15)];
    dv_record = [dv_record; dv(14:15)];
    ttci_record = [ttci_record; ttci(14:15)];
    da_record = [da_record; da(14:15)];

    % scatter(ds(13:15), thwi(13:15), 30, dv(13:15), 'filled');hold on
    % xlabel('Distance headway [m]')
    % ylabel('Time headway Inverse [s^{-1}]')
    % grid on
    % c = colorbar;

    % c.Label.String = 'Intervention duration [s]';

    % subplot(2,1,2)
    % scatter(dv(1:2), ttci(1:2), 30, 'filled');hold on
    % grid on
    % c = colorbar;
    % c.Label.String = 'Intervention duration [s]';
end