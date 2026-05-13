%%  Speed profile processing scripts
clear
clc
close all
dbstop if error
save_data = input('Do you wish to save this data [0 & 1]: ');

%%  Choose and load the speed profile data
filename = uigetfile('.csv', 'Choose driving cycle file.');
data = load(filename);

t = data(:, 1);
v = data(:, 2);
a = data(:, 3);
s = data(:, 4);

%%  Generate scaled speed profile
dt = mean(diff(t));
aa = a*1.2;
vv = cumsum(aa * dt);
ss = cumsum(0.5 * (vv(2:end) + vv(1:end-1)) * dt);
ss = [ss; ss(end)];

fig = figure(1);
subplot(3,1,1)
plot(t, a, t, aa,'LineWidth',2);
xlabel('Time [s]')
ylabel('Acceleration [m/s^{2}]')

subplot(3,1,2)
plot(t, v, t, vv,'LineWidth',2);
xlabel('Time [s]')
ylabel('Speed [m/s]')

subplot(3,1,3)
plot(t, s, t, ss,'LineWidth',2);
xlabel('Time [s]')
ylabel('Distance [m]')

%%  Clip the driving cycle
figure(fig);
selection_confirmed = false;
while ~selection_confirmed
    disp('Select the start and end times from figure(1).');
    [clip_t_selected, ~] = ginput(2);
    clip_t_selected = sort(clip_t_selected);

    [~, start_t_id] = min(abs(t - clip_t_selected(1)));
    [~, end_t_id] = min(abs(t - clip_t_selected(2)));

    figure(fig);
    subplot(3,1,1)
    hold on
    selected_acc_markers = plot( ...
        t([start_t_id, end_t_id]), ...
        aa([start_t_id, end_t_id]), ...
        'ro', 'MarkerSize', 8, 'LineWidth', 2);
    hold off

    subplot(3,1,2)
    hold on
    selected_spd_markers = plot( ...
        t([start_t_id, end_t_id]), ...
        vv([start_t_id, end_t_id]), ...
        'ro', 'MarkerSize', 8, 'LineWidth', 2);
    hold off

    subplot(3,1,3)
    hold on
    selected_dist_markers = plot( ...
        t([start_t_id, end_t_id]), ...
        ss([start_t_id, end_t_id]), ...
        'ro', 'MarkerSize', 8, 'LineWidth', 2);
    hold off

    confirm_msg = sprintf([ ...
        'Start point\\n' ...
        '  Time: %.2f s\\n' ...
        '  Speed: %.2f m/s\\n' ...
        '  Acceleration: %.2f m/s^2\\n\\n' ...
        'End point\\n' ...
        '  Time: %.2f s\\n' ...
        '  Speed: %.2f m/s\\n' ...
        '  Acceleration: %.2f m/s^2\\n\\n' ...
        'Do you want to use this selection?'], ...
        t(start_t_id), vv(start_t_id), aa(start_t_id), ...
        t(end_t_id), vv(end_t_id), aa(end_t_id));

    selection_answer = questdlg(confirm_msg, ...
        'Confirm Driving Cycle Clip', ...
        'Use Selection', 'Reselect', 'Use Selection');
    selection_confirmed = strcmp(selection_answer, 'Use Selection');

    if ~selection_confirmed
        delete(selected_acc_markers);
        delete(selected_spd_markers);
        delete(selected_dist_markers);
    end
end

t_clip = t(start_t_id:end_t_id);
vv_clip = vv(start_t_id:end_t_id);
aa_clip = aa(start_t_id:end_t_id);
ss_clip = ss(start_t_id:end_t_id);

t_clip = t_clip - t_clip(1);
ss_clip = ss_clip - ss_clip(1);

figure(2);
subplot(3,1,1)
plot(t_clip, aa_clip,'LineWidth',2);
xlabel('Time [s]')
ylabel('Acceleration [m/s^{2}]')

subplot(3,1,2)
plot(t_clip, vv_clip,'LineWidth',2);
xlabel('Time [s]')
ylabel('Speed [m/s]')

subplot(3,1,3)
plot(t_clip, ss_clip,'LineWidth',2);
xlabel('Time [s]')
ylabel('Distance [m]')

%%  Save the speed profile
if save_data
    scenario_name = input('Please enter scenario name: ','s');
    data_to_save = [t_clip, vv_clip, aa_clip, ss_clip];
    writematrix(data_to_save, strcat(scenario_name, '.csv'));
end
