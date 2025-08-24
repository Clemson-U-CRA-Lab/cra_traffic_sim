%% Optimizing map file for AR visualization
clear
clc
close all
dbstop if error

save_data = true;
%%  Load the map
dir0_filename = "custom_map_dir0.csv";
dir1_filename = "custom_map_dir1.csv";
map_origin_filename = "custom_map_origin.csv";

%%  Define parameters
track_length = input('Please input track length in meters: ');
track_width = input('Please input track width in meters: ');

%%  Generate map files
map_origin = [0, 0, 0, track_length;
              track_width, 0, track_width, track_length];

dir0_x = 0:0.1:track_length;
dir0_y = zeros(1, length(dir0_x));
dir0_z = zeros(1, length(dir0_x));
dir0_yaw = zeros(1, length(dir0_x));
dir0_pitch = zeros(1, length(dir0_x));
dir0_s = dir0_x;

dir1_x = track_length:-0.1:0;
dir1_y = ones(1, length(dir0_x)) * track_width;
dir1_z = zeros(1, length(dir0_x));
dir1_yaw = ones(1, length(dir0_x)) * pi;
dir1_pitch = zeros(1, length(dir0_x));
dir1_s = dir0_x;

%%  Record maps
data_dir0 = [dir0_x.', dir0_y.', dir0_z.', dir0_yaw.', dir0_pitch.', dir0_s.'];
data_dir1 = [dir1_x.', dir1_y.', dir1_z.', dir1_yaw.', dir1_pitch.', dir1_s.'];

pdir = fileparts(pwd);
origin_dir = strcat([pdir, '/scripts/map_origins/']);

if save_data
    writematrix(data_dir0, 'custom_map_dir0.csv');
    writematrix(data_dir1, 'custom_map_dir1.csv');
    writematrix(map_origin, strcat([origin_dir, 'custom_map_endpoint.csv']));
end