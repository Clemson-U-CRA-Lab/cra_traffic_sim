import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def TTCi_estimate(ego_v, front_v, front_s):
    ttc_i = (ego_v - front_v) / front_s
    return ttc_i

class IDM():
    def __init__(self, a, b, s0, v0, T):
        self.a = a
        self.b = b
        self.s0 = s0
        self.v0 = v0
        self.T = T

    def IDM_acceleration(self, front_v, ego_v, front_s, ego_s):
        s_safe = self.s0 + ego_v * self.T + front_v * (ego_v - front_v) / (2 * (self.a * self.b)**0.5)
        # s_safe[s_safe < self.s0 + 3] = self.s0 + 3
        acc = self.a * (1 - (ego_v / self.v0) ** 4 -
                        (s_safe / (front_s - ego_s - self.s0)) ** 2)
        acc = np.clip(acc, -4, 4)
        return [acc]

class Model(nn.Module):
    def __init__(self, in_features=3, h1=256, h2=256, h3=32, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.out(x)
        return x

class Model_3_input(nn.Module):
    def __init__(self, in_features=4, h1=256, h2=256, h3=32, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.out(x)
        return x

class NN_controller():
    def __init__(self, nn_pt_file, input_num):
        self.num_input = input_num
        if input_num == 3:
            self.nn_controller = Model(h1=256, h2=256)
        if input_num == 4:
            self.nn_controller = Model_3_input(h1=256, h2=256)
        self.nn_controller.eval()
        self.nn_controller.load_state_dict(torch.load(nn_pt_file, map_location='cpu'))
        self.nn_controller.to('cuda')
        self.IDM_brake = IDM(a=3, b=3, s0=5, v0=20, T=5)
    
    def step_forward(self, s_vt, pv_vt, s_st, pv_st, s_at, pv_at, use_prediction_horizon, sim_t):
        ttc_i = TTCi_estimate(ego_v=s_vt, front_v=pv_vt, front_s=pv_st - s_st)
        # Calculate the prediction horizon length
        pv_s_end = np.zeros(1)
        pv_v_end = np.zeros(1)
        a_input = np.array(pv_at)
        a = np.tile(a_input, (49, 1))
        v = pv_vt + np.cumsum(a * 0.5, axis=0)
        v = np.clip(v, 0, np.Inf)
        s = pv_st + np.cumsum(v * 0.5, axis=0)
        pv_s_end = np.clip(s[-1, :] - s_st, -10, 1500)
        pv_v_end = v[-1, :] - s_vt
        sig_dv = 25
        sig_ds = 300
        c_dv = 55.13
        c_ds = 910
        
        if self.num_input == 3:
            if use_prediction_horizon:
                # nn_input_vec = np.array([s_vt, 
                #                          np.exp(-(pv_v_end - c_dv) / (2 * sig_dv)), 
                #                          np.exp(-(pv_s_end - c_ds) / (2 * sig_ds))])
                nn_input_vec = np.array([s_vt, pv_v_end[0], pv_s_end[0]])
            else:
                nn_input_vec = np.array([s_vt, pv_vt - s_vt, pv_st - s_st])
        if self.num_input == 4:
            nn_input_vec = np.array([s_vt, pv_vt - s_vt, pv_s_end, pv_v_end])
        nn_input = torch.FloatTensor(nn_input_vec.T).cuda()
        # Compute the neural network control
        with torch.no_grad():
            ego_a_nn = self.nn_controller.forward(nn_input)
            s_a_nn = ego_a_nn.flatten().tolist()
            
        # Compute intelligent driver model control
        # s_a_IDM = self.IDM_brake.IDM_acceleration(front_v=pv_vt, ego_v=s_vt, front_s=pv_st, ego_s=s_st)
        
        # # Check the if IDM braking is needed
        # if len(s_a_IDM) > 0:
        #     det = ((ttc_i > 0.25) * (pv_st - s_st < 15)).astype(bool)
        #     IDM_w = det.astype(float)
        #     ego_a_tgt = IDM_w * s_a_IDM + (1.0 - IDM_w) * s_a_nn
        # else:
        #     ego_a_tgt = None
            
        return ego_a_nn.flatten().tolist()[0]
