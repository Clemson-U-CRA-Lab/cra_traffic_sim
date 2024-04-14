%% Optimizing map file for AR visualization
clear
clc
close all
dbstop if error

save_data = true;
%%  Load the map
map_filename = "CMI_outerloop_mach_e_wider_lane.csv";
map_data = load(map_filename);

x = map_data(:, 1);
y = map_data(:, 2);
z = map_data(:, 3);
yaw = map_data(:, 4);
pitch = map_data(:, 5);
dist = map_data(:, 6);

% pitch estimated 
dxdy = (diff(x).^2 + diff(y).^2).^0.5;
dz = diff(z);
pitch_esti = smoothdata(atan(dz./ dxdy), 'gaussian', 20);
pitch_esti = [pitch_esti; pitch_esti(end)];

figure(1)
subplot(2,1,1)
plot(pitch);hold on
plot(pitch_esti)
legend('Pitch', 'Pitch Estimated')
subplot(2,1,2)
plot(z)

if save_data
    writematrix([x,y,z,yaw,pitch_esti,dist], map_filename);
end