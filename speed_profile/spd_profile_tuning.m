%%  Speed profile processing scripts
clear
clc
close all
dbstop if error
save_data = input('Do you wish to save this data [0 & 1]: ');

%%  Choose and load the speed profile data
filename = 'ITIC_Lane_Change_Modeling_StartLane_Fast.csv';
data = load(filename);

t = data(:, 1);
v = data(:, 2);
a = data(:, 3);
s = data(:, 4);

%%  Generate scaled speed profile
dt = mean(diff(t));
aa = 0.33 * a;
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

%%  Save the speed profile
if save_data
    data_to_save = [t, vv, aa, ss];
    writematrix(data_to_save, 'CMI_simple_scenario.csv');
end