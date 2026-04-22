clear
clc
close all
dbstop if error

%%  Configuration
default_lift_method = "edmd_ttci_thwi"; % Options: "edmd_ttci_thwi", "sindy_ttci_thwi"
lift_method = default_lift_method;

% Show a lift-method chooser when MATLAB GUI is available.
if usejava('desktop') && usejava('awt')
    selection = questdlg( ...
        'Choose the lifting method:', ...
        'Lift Method', ...
        'EDMD + TTCI/THWI', ...
        'SINDy + TTCI/THWI', ...
        'EDMD + TTCI/THWI');

    if strcmp(selection, 'EDMD + TTCI/THWI')
        lift_method = "edmd_ttci_thwi";
    elseif strcmp(selection, 'SINDy + TTCI/THWI')
        lift_method = "sindy_ttci_thwi";
    elseif isempty(selection)
        lift_method = default_lift_method;
    else
        error('Unsupported lift-method selection.');
    end
end

%%  Section 1: Load the reward data
data = readtable("dc1_reward_trajectory.csv");
ds = data.space_error.';
dv = data.speed_error.';
v_ego = data.ego_speed.';
v_front = data.ego_speed.' + data.speed_error.';
a_front = data.front_acceleration.';
a_ego = data.ego_acceleration.';
reward = data.reward.';
t = 0:0.1:0.1*(length(ds)-1);

% Guard against divide-by-zero in inverse-headway features.
ds_safe = max(ds, 1e-3);
ttci = dv ./ ds_safe;
thwi = v_ego ./ ds_safe;

%%  Section 2: Lift the space using EDMD
c_ds = linspace(5, 50, 2);
c_dv = linspace(-10, 10, 2);
c_ttci = linspace(-0.5, 0.5, 2);
c_thwi = linspace(0, 2.5, 2);
sigma = 0.5;

% Construct lifting space
z_baseline = koopman_lift_sindy_baseline(ds, dv);
z = build_lifted_state(ds, dv, ttci, thwi, lift_method, c_ds, c_dv, c_ttci, c_thwi, sigma);

%%  Quick comparison between the baseline 4-state lift and the richer 6-state lift
C_baseline = reward * pinv(z_baseline);
x_esti_baseline = C_baseline * z_baseline;
baseline_rmse = sqrt(mean((reward - x_esti_baseline).^2));
baseline_r2 = 1 - sum((reward - x_esti_baseline).^2) / sum((reward - mean(reward)).^2);

%%  Compuate C matrices
C = reward * pinv(z);
Y = z(:, 2:end);
X = z(:, 1:end-1);
U = a_front(1:end-1) - a_ego(1:end-1);
M = Y * pinv([X; U]);
A = M(:, 1:end-1);
B = M(:, end:end);

%%  Section 3: Some quick sanity checks
x_esti = C*z;
x_esti(x_esti > 60)=60; x_esti(x_esti < 0)=0;
x_esti_baseline(x_esti_baseline > 60)=60; x_esti_baseline(x_esti_baseline < 0)=0;

extended_rmse = sqrt(mean((reward - x_esti).^2));
extended_r2 = 1 - sum((reward - x_esti).^2) / sum((reward - mean(reward)).^2);

disp(['Selected lift method: ', char(lift_method)]);
disp(['Baseline 4-state lift RMSE: ', num2str(baseline_rmse), ', R^2: ', num2str(baseline_r2)]);
disp(['Selected lift RMSE: ', num2str(extended_rmse), ', R^2: ', num2str(extended_r2)]);

plot(t, reward, '-k', 'LineWidth', 3);hold on
plot(t, smoothdata(x_esti_baseline, 'rlowess', 10), '--b', 'LineWidth', 2)
plot(t, smoothdata(x_esti, 'rlowess', 10), '-r', 'LineWidth', 3)
xlabel('Time [s]');ylabel('Reward');xlim([0 250])
legend('Assgined reward to human intervention', ...
       'Estimated reward from baseline 4-state lift', ...
       ['Estimated reward from selected lift: ', char(lift_method)])

%% Use LQR to sanity check the result
z_tgt = max(x_esti)*C';
u_ss = pinv(B)*(eye(length(z_tgt)) - A)*z_tgt;
ds_sample = 10;
dv_sample = 5;
v_ego_sample = 0;
v_front_sample = v_ego_sample + dv_sample;

ttci_sample = dv_sample / max(ds_sample, 1e-3);
thwi_sample = v_ego_sample / max(ds_sample, 1e-3);
z_t = build_lifted_state(ds_sample, dv_sample, ttci_sample, thwi_sample, lift_method, c_ds, c_dv, c_ttci, c_thwi, sigma);

Q_z = C.'*1*C;
R = 1;
P = Q_z;

for N = 5:-1:1
    K = (R + B.'*P*B)^(-1)*B.'*P*A;
    u = -K*(z_t - z_tgt);
    P = Q_z + A.'*P*A - A.'*P*B*K;
end

%%  Save the matrix
save_data = input('Do you wish to save data?[0 or 1]: ');
if save_data
    [A_m, A_n] = size(A);
    [B_m, B_n] = size(B);
    [C_m, C_n] = size(C);
    writematrix(A, strcat(['A_', num2str(A_m), 'x', num2str(A_n), '_matrix.csv']));
    writematrix(B, strcat(['B_', num2str(B_m), 'x', num2str(B_n), '_matrix.csv']));
    writematrix(C, strcat(['C_', num2str(C_m), 'x', num2str(C_n), '_matrix.csv']));
end

%%  Add lifting function
function z = koopman_lift_edmd(ds, dv, ttci, thwi, c_ds, c_dv, c_ttci, c_thwi, sigma)
    z = [];
    for i=1:1:length(c_ds)
        for j=1:1:length(c_dv)
            for k=1:1:length(c_ttci)
                for m=1:1:length(c_thwi)
                    phi = exp(-vecnorm([ds; dv; ttci; thwi] - [c_ds(i); c_dv(j); c_ttci(k); c_thwi(m)] / (2 * sigma^2)));
                    z = [z; phi];
                end
            end
        end
    end
end

function z = koopman_lift_sindy_baseline(ds, dv)
    z = [ds; dv; ds.^2; dv.^2];
end

function z = koopman_lift_sindy_ttci_thwi(ds, dv, ttci, thwi)
    z = [ds; dv; ds.^2; dv.^2; ttci; thwi];
end

function z = build_lifted_state(ds, dv, ttci, thwi, lift_method, c_ds, c_dv, c_ttci, c_thwi, sigma)
    if lift_method == "edmd_ttci_thwi"
        z = koopman_lift_edmd(ds, dv, ttci, thwi, c_ds, c_dv, c_ttci, c_thwi, sigma);
    elseif lift_method == "sindy_ttci_thwi"
        z = koopman_lift_sindy_ttci_thwi(ds, dv, ttci, thwi);
    else
        error("Unsupported lift_method. Use ""edmd_ttci_thwi"" or ""sindy_ttci_thwi"".");
    end
end

function a = IDM(ds, v, dv)
    s_star = 5 + v * 1.5 + v*dv / (2*sqrt(2*2));
    a = 3 * (1 - (v / 20)^4 - (s_star / ds)^2);
end
