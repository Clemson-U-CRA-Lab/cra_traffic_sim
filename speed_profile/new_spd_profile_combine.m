clear
clc
close all
dbstop if error

%%  Choose and load the speed profile data
filename0 = uigetfile('.csv', 'Choose 1st driving cycle file.');
filename1 = uigetfile('.csv', 'Choose 2nd driving cycle file.');
filename2 = uigetfile('.csv', 'Choose 3rd driving cycle file.');

data0 = load(filename0);
data1 = load(filename1);
data2 = load(filename2);

%%  Combine driving cycle
data = struct("data0", data0, "data1", data1, "data2", data2);
t = [0];
spd = [0];
acc = [0];
dist = [0];

for i = 0:1:2
    data_name = strcat(['data', num2str(i)]);
    data_selected = data.(data_name);
    t = [t; data_selected(:, 1) + t(end)];
    spd = [spd; data_selected(:, 2)];
    acc = [acc; data_selected(:, 3)];
    dist = [dist; data_selected(:, 4) + dist(end)];
end

figure(1)
subplot(2,1,1)
plot(t, spd);
subplot(2,1,2)
plot(t, acc)

%%  Save the data
data_to_save = [t, spd, acc, dist];
new_filename = input('Please enter speed profile name: ','s');
writematrix(data_to_save, strcat([new_filename, '.csv']));