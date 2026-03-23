human_intervention_analysis

%%  Scatter the ttci profile
dt = mean(diff(sim_t));
prefered_ttci = 0.2105 ./ (1 + exp(0.1102 * (front_s - ego_s - 14.06)));
ttci = (front_v - ego_v)./(front_s - ego_s);

d_prefered_ttci = smoothdata(diff(prefered_ttci)/dt, 'rlowess', 25);
d_ttci = smoothdata(diff(ttci)/dt, 'rlowess', 25);
d_prefered_ttci = [d_prefered_ttci; d_prefered_ttci(end)];
d_ttci = [d_ttci; d_ttci(end)];

gap_ttci = abs(prefered_ttci - ttci);
gap_d_ttci = abs(d_prefered_ttci - d_ttci);

figure(5)
subplot(2,1,1)
plot(sim_t, ttci, 'LineWidth', 2);hold on
plot(sim_t, prefered_ttci, 'LineWidth', 2);
legend('Baseline MPC','Human-intervened data generated');
xlabel('Time [s]');
ylabel('Time-to-collision [s^{-1}]');
% plot(sim_t, gap_d_ttci);

subplot(2,1,2)
plot(sim_t, front_v, 'LineWidth', 2);hold on
plot(sim_t, ego_v, 'LineWidth', 2);
xlabel('Time [s]');
ylabel('Speed [m/s]');
%%  Save clip from driving
while true
    save_clip = input('Do you want to save any clip? [1 or 0] ');
    if ~save_clip
        break;
    else
        start_t = input('Select start time [s]: ');
        end_t = input('Select end time [s]: ');
        [~, start_t_id] = min(abs(start_t - sim_t));
        [~, end_t_id] = min(abs(end_t - sim_t));
        front_a_clip = front_a(start_t_id:end_t_id);
        front_v_clip = front_v(start_t_id:end_t_id);
        sim_t_clip = sim_t(start_t_id:end_t_id) - sim_t(start_t_id);
        
        data_header = {'time', 'acceleration', 'velocity'};
        data_to_save = [sim_t_clip, front_a_clip, front_v_clip];
        T = array2table(data_to_save, 'VariableNames', data_header);

        clip_name_id = input('Choose clip id: ');
        clip_name = strcat(['dc_clip_', num2str(clip_name_id), '.csv']);
        writetable(T, clip_name);

        clc;
    end
end