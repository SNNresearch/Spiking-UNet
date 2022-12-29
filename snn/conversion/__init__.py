import torch
import torch.nn as nn
import copy
from .. import spiking_neuron as neuron
from .model_lib import pytorch_kernel as kernel
from .model_lib import pytorch_kernel_fx as kernel_fx
import os

class Parser:
    def __init__(self, path = None):
        self.v_threshold = 1.0
        self.v_reset = 0.0
        self.path = path
        # self.task = task
        if not os.path.exists(self.path):
            os.makedirs(self.path)


    def parse(self, model, norm_data, method: str, scale_method: str) -> nn.Module:

        if method == 'layer_wise':
            torch_kernel = kernel.torch_kernel(self.path, method, scale_method)
            model.to('cpu')
            norm_data = norm_data.to('cpu')
            model = torch_kernel.weight_normalization(model, norm_data)
        elif method == 'connection_wise':
            torch_kernel = kernel_fx.torch_kernel(self.path, method, scale_method)
            model = torch_kernel.weight_normalization(model, norm_data)
        else:
            pass

        return model
    
    def convert_to_snn(self, modules, neuron_class, timesteps, reset_method, v_threshold=1.0):
        for n, module in modules.named_children():
            if len(list(module.children())) > 0:
                modules = self.convert_to_snn(module, neuron_class, stbp, timesteps, reset_method, v_threshold)
            else:
                if isinstance(module, nn.ReLU):
                    if neuron_class == 'multi':
                        new_layer = neuron.Multi_Threshold_Neuron(v_threshold=1.0, v_reset=0.0, state='eval')
                    else:
                        new_layer = neuron.IF_Neuron(v_threshold=v_threshold, v_reset=0, state='eval', reset_method=reset_method, timesteps=timesteps)
                    setattr(modules, n, new_layer)

                if isinstance(module, nn.AvgPool2d):
                    if neuron_class == 'multi':
                        new_layer =  nn.Sequential(module, neuron.Multi_Threshold_Neuron(v_threshold=1.0, v_reset=0.0, state='eval'))
                    else:
                        new_layer = nn.Sequential(module, neuron.IF_Neuron(v_threshold=v_threshold, v_reset=0.0, state='eval', reset_method=reset_method, timesteps=timesteps))
                    setattr(modules, n, new_layer)
        return modules
