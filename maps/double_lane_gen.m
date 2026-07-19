%%  Generate ITIC-style double-lane maps from two physical lane maps
clear
clc
close all
dbstop if error

%%  Section 1. Configuration
% Input maps must use columns: x, y, z, yaw, pitch, s.
input_lane1_file = "custom_map_dir0.csv";
input_lane2_file = "custom_map_dir1.csv";
output_prefix = "custom_map";

% Configure each input lane so both dir0 outputs travel the same direction.
reverse_lane1_for_dir0 = false;
reverse_lane2_for_dir0 = true;

save_data = true;
show_figures = true;

%%  Section 2. Load and validate map data
map_1 = load(input_lane1_file);
map_2 = load(input_lane2_file);

validate_map(map_1, input_lane1_file);
validate_map(map_2, input_lane2_file);

%%  Section 3. Build lane-numbered maps for both directions
dir0_lane1 = orient_map(map_1, reverse_lane1_for_dir0);
dir0_lane2 = orient_map(map_2, reverse_lane2_for_dir0);

dir1_lane1 = reverse_map(dir0_lane1);
dir1_lane2 = reverse_map(dir0_lane2);

%%  Section 4. Visualize source and generated maps
if show_figures
    figure(1)
    scatter(map_1(:, 1), map_1(:, 2), 20, map_1(:, 6), 'filled');hold on
    scatter(map_2(:, 1), map_2(:, 2), 20, map_2(:, 6), 'filled');hold off
    title('Source Maps')
    xlabel('X [m]')
    ylabel('Y [m]')
    legend('Input Lane 1',  'Input Lane 2')
    axis equal
    grid on
    colorbar

    plot_generated_maps(dir0_lane1, dir0_lane2, 'Generated Direction 0')
    plot_generated_maps(dir1_lane1, dir1_lane2, 'Generated Direction 1')
end

%%  Section 5. Save generated maps
if save_data
    writematrix(dir0_lane1, output_prefix + "_dir0_lane1.csv");
    writematrix(dir0_lane2, output_prefix + "_dir0_lane2.csv");
    writematrix(dir1_lane1, output_prefix + "_dir1_lane1.csv");
    writematrix(dir1_lane2, output_prefix + "_dir1_lane2.csv");
end

%%  Helper functions
function validate_map(map_data, map_file)
    if size(map_data, 2) ~= 6
        error('Map file %s must have exactly 6 columns: x, y, z, yaw, pitch, s.', char(map_file));
    end

    if size(map_data, 1) < 2
        error('Map file %s must contain at least two rows.', char(map_file));
    end
end

function map_out = orient_map(map_data, reverse_direction)
    map_out = normalize_s(map_data);

    if reverse_direction
        map_out = reverse_map(map_out);
    end
end

function map_out = reverse_map(map_data)
    map_out = flipud(map_data);
    s_end = map_data(end, 6);
    map_out(:, 4) = wrap_angle(map_out(:, 4) + pi);
    map_out(:, 5) = -map_out(:, 5);
    map_out(:, 6) = s_end - map_out(:, 6);
    map_out = normalize_s(map_out);
end

function map_out = normalize_s(map_data)
    map_out = map_data;
    map_out(:, 6) = map_out(:, 6) - map_out(1, 6);

    if any(diff(map_out(:, 6)) < 0)
        error('Map distance column must increase after orientation. Check reverse configuration.');
    end
end

function yaw_wrapped = wrap_angle(yaw)
    yaw_wrapped = mod(yaw + pi, 2 * pi) - pi;
end

function plot_generated_maps(lane1, lane2, plot_title)
    figure
    plot_lane_with_endpoints(lane1, 'b', 'Lane 1')
    hold on
    plot_lane_with_endpoints(lane2, 'r', 'Lane 2')
    hold off
    title(plot_title)
    xlabel('X [m]')
    ylabel('Y [m]')
    legend('Lane 1 Path', 'Lane 1 Start', 'Lane 1 End', ...
           'Lane 2 Path', 'Lane 2 Start', 'Lane 2 End')
    axis equal
    grid on
end

function plot_lane_with_endpoints(lane_map, lane_color, display_name)
    path_name = char(string(display_name) + " Path");
    start_name = char(string(display_name) + " Start");
    end_name = char(string(display_name) + " End");

    plot(lane_map(:, 1), lane_map(:, 2), lane_color, 'DisplayName', path_name)
    scatter(lane_map(1, 1), lane_map(1, 2), 80, lane_color, 'filled', ...
            'DisplayName', start_name)
    scatter(lane_map(end, 1), lane_map(end, 2), 80, lane_color, ...
            'DisplayName', end_name)
end
