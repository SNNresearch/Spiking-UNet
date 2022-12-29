import torch
import torch.nn as nn

def grad_cal(scale, IF_in):
    out = scale * IF_in.gt(0).type(torch.cuda.FloatTensor)
    return out

class Multi_Threshold_Neuron(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, state='eval'):
        super(Multi_Threshold_Neuron, self).__init__()
        self.state = state
        self.register_buffer('vth_half_half', torch.tensor(0.25))
        self.register_buffer('vth_half', torch.tensor(0.5))
        self.register_buffer('vth', torch.tensor(v_threshold))
        self.register_buffer('vth_high', torch.tensor(2.0))
        self.register_buffer('v_reset', torch.tensor(v_reset))
        self.register_buffer('v', torch.tensor(v_reset))
        self.register_buffer('acc_input', torch.tensor(v_reset))
        self.register_buffer('output', torch.tensor(v_reset))
        self.register_buffer('spikecount', torch.tensor(v_reset))
    
    def forward(self, dv):
        if self.state == 'train':
            with torch.enable_grad():
                input = dv
                tem = self.output.detach()
                v_th = self.vth.broadcast_to(input.size())
                scale = 1.0
                self.scale = grad_cal(scale, self.acc_input)
                out = torch.mul(self.scale, input)
                total_output = out - out.detach() + tem 
                return total_output

        else:
            self.v = self.v + dv
            spike_v2 = torch.gt(self.v, self.vth_high) 
            self.v[spike_v2] = self.v[spike_v2] - self.vth_high
            spike_v1 = torch.gt(self.v, self.vth)
            self.v[spike_v1] = self.v[spike_v1] - self.vth
            spike_half = torch.gt(self.v, self.vth_half)
            self.v[spike_half] = self.v[spike_half] - self.vth_half
            spike_half_half = torch.gt(self.v, self.vth_half_half)
            self.v[spike_half_half] = self.v[spike_half_half] - self.vth_half_half

            spike_total = self.vth_high * spike_v2 + self.vth * spike_v1 + self.vth_half * spike_half + self.vth_half_half * spike_half_half

            self.acc_input = self.acc_input + dv
            self.output = self.output + spike_total
            self.spikecount = self.spikecount + spike_v2 + spike_v1 + spike_half + spike_half_half

            return spike_total

    def set_state(self, state):
        self.state = state
        
    def reset(self):
        self.v = self.v_reset
        self.acc_input = self.v_reset
        self.output = self.v_reset
        self.spikecount = self.v_reset

class IF_Neuron(nn.Module):
    def __init__(self, v_threshold = 1.0, v_reset = 0.0, state = 'eval', reset_method = 'reset_by_subtraction', timesteps = 10):
        super().__init__()
        self.register_buffer('vth', torch.tensor(v_threshold))
        self.register_buffer('v_reset', torch.tensor(v_reset))
        self.register_buffer('v', torch.tensor(v_reset))
        self.register_buffer('acc_input', torch.tensor(v_reset))
        self.register_buffer('output', torch.tensor(v_reset))
        self.register_buffer('residual_v', torch.tensor(v_reset))
        self.register_buffer('spikecount', torch.tensor(v_reset))
        self.reset_method = reset_method
        self.state = state
        self.timesteps = timesteps
    
    def forward(self, dv):
        if self.state == 'train':
            input = dv
            tem = self.output.detach()
            v_th = self.vth.broadcast_to(input.size())
            scale = 1.0
            self.scale = grad_cal(scale, self.acc_input)
            out = torch.mul(self.scale, input)
            total_output = out - out.detach() + tem 
            return total_output

        else:
            self.v = self.v + dv
            spike = torch.ge(self.v, self.vth)
            
            self.acc_input = self.acc_input + dv
            self.output = self.output + spike

            if self.reset_method == 'reset_by_subtraction':
                self.v[spike > 0] = self.v[spike > 0] - self.vth
            else:
                self.v[spike > 0] = self.v_reset.float()

            self.spikecount = self.spikecount + spike

            spike_total = self.vth * spike

            return spike_total

    
    def set_state(self, state):
        self.state = state
    
    def reset(self):
        self.v = self.v_reset
        self.acc_input = self.v_reset
        self.output = self.v_reset
        self.spikecount = self.v_reset

