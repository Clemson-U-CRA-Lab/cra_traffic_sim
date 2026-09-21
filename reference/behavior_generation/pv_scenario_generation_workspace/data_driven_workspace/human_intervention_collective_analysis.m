clear
clc
close all
dbstop if error

%% Input
new_input = true;
ds_record = [];
dv_record = [];
da_record = [];
ttci_record = [];

while true
    new_input = input('Do you want to add new test result? [0 or 1]: ');
    if new_input == 0
        break
    else
        human_intervention_analysis_logged_data;
    end
end

%%  Check the overall performance using curve fitting
% Find only pulling scenarios
dv_record_id = find(dv_record > 0);
dv_record = dv_record(dv_record_id);
ds_record = ds_record(dv_record_id);
ttci_record = ttci_record(dv_record_id);
da_record = da_record(dv_record_id);

figure(5)
scatter3(dv_record, ds_record, ttci_record, 50, "filled");

%%  Curve fit the data
[f, gof, output] = fit(ds_record, ttci_record, 'logistic');
figure(6)
scatter(ds_record, ttci_record, 50, dv_record,"filled"); hold on
plot(f);
c = colorbar;
c.Label.String = 'Speed gap [m/s]';
c.Location = 'southoutside';
grid on
xlabel('Distance headway [m]')
ylabel('Time-to-collision Inverse [s^{-1}]')
legend(strcat(['R-Square: ', num2str(gof.rsquare)]))