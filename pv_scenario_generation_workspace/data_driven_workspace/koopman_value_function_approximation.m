clear
clc
close all
dbstop if error

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

%%  Section 2: Lift the space using EDMD
c_ds = linspace(5, 50, 10);
c_dv = linspace(-10, 10, 10);
c_v_ego = linspace(0, 20, 10);
sigma = 0.5;

% Construct lifting space
% z = koopman_lift_edmd(ds, dv, v_ego, c_ds, c_dv, c_v_ego, sigma);
z = koopman_lift_sindy(ds, dv, v_ego, v_front);

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
plot(t, reward, '-k', 'LineWidth', 3);hold on
plot(t, smoothdata(x_esti, 'rlowess', 10), '-r', 'LineWidth', 3)
xlabel('Time [s]');ylabel('Reward');xlim([0 250])
legend('Assgined reward to human intervention','Estimated reward from value function in higher space')

%% Use LQR to sanity check the result
z_tgt = max(x_esti)*C';
u_ss = pinv(B)*(eye(length(z_tgt)) - A)*z_tgt;
ds_sample = 10;
dv_sample = 5;
v_ego_sample = 0;
v_front_sample = v_ego_sample + dv_sample;

% z_t = koopman_lift_edmd(25, -5, c_ds, c_dv, sigma);
z_t = koopman_lift_sindy(ds_sample, dv_sample, v_ego_sample, v_front_sample);

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
function z = koopman_lift_edmd(ds, dv, v_ego, c_ds, c_dv, c_v_ego, sigma)
    z = [];
    for i=1:1:length(c_ds)
        for j=1:1:length(c_dv)
            for k=1:1:length(c_v_ego)
                phi = exp(-vecnorm([ds; dv; v_ego] - [c_ds(i); c_dv(j); c_v_ego(k)] / (2 * sigma^2)));
                z = [z; phi];
            end
        end
    end
end

function z = koopman_lift_sindy(ds, dv, v_ego, a_ego)
%     z = [ds; dv; v_ego; a_ego; ...
%          ds.^2; dv.^2; v_ego.^2; a_ego.^2; ...
%          ds.*dv; ds.*v_ego; ds.*a_ego;...
%          dv.*v_ego;  dv.*a_ego; v_ego.*a_ego];
    z = [ds; dv; ds.^2; dv.^2];
end

function a = IDM(ds, v, dv)
    s_star = 5 + v * 1.5 + v*dv / (2*sqrt(2*2));
    a = 3 * (1 - (v / 20)^4 - (s_star / ds)^2);
end