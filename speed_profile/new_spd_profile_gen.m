clear
clc
close all
dbstop if error

%%  Choose and load the speed profile data
filename = uigetfile('.csv', 'Choose driving cycle file.');
filename_split = split(filename,'.');
spd_profile_name = filename_split{1};
data = load(filename);

t = data(:, 1);
v = data(:, 2);
a = data(:, 3);
s = data(:, 4);

dt = mean(diff(t));
%%  Sanity check the data
figure(1)
subplot(3,1,1)
plot(t, v);
subplot(3,1,2)
plot(t, a)
subplot(3,1,3)
plot(t, s)

%%  Generate new spd profile from acceleration data
vv = smoothdata(v, 'gaussian', 50);
aa = [0; diff(vv)./dt];
ss = cumsum(vv * dt);

figure(1)
subplot(3,1,1)
plot(t, vv);
subplot(3,1,2)
plot(t, aa)
subplot(3,1,3)
plot(t, ss)

%%  Find start and end time with velocity modification
start_t = input('Please enter start time: ');
end_t = input('Please enter end time: ');
w = input('Please enter speed clip ratio: ');

[~, start_t_id] = min(abs(start_t - t));
[~, end_t_id] = min(abs(end_t - t));

v_clip = vv(start_t_id:end_t_id) * w;
a_clip = [0; diff(v_clip)/dt];
s_clip = cumsum(v_clip * dt);
t_clip = t(start_t_id:end_t_id) - t(start_t_id);

figure(1)
subplot(3,1,1)
plot(t_clip, v_clip / 0.447);
subplot(3,1,2)
plot(t_clip, a_clip)
subplot(3,1,3)
plot(t_clip, s_clip)

%%  Save the data
data_to_save = [t_clip, v_clip, a_clip, s_clip];
writematrix(data_to_save, strcat([spd_profile_name,'_sec_2.csv']));