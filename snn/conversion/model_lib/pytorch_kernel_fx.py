import os
import torch
import numpy as np
import torch.nn as nn
from torch.fx import Interpreter


class TorchKernelInterpreter(Interpreter):
    def __init__(self, mod: torch.nn.Module):
        gm = torch.fx.symbolic_trace(mod, concrete_args={'input_type': 'original'})
        super().__init__(gm)
        self.lambda_input = {}
        self.lambda_output = {}
        self.shape_dict = {}

    def run_node(self, n: torch.fx.Node):
        output = super().run_node(n)
        if n.op in ['call_module', 'output']:
            if 'relu' in n.target.lower() or n.op == 'output':
                if isinstance(output, torch.Tensor):
                    self.shape_dict[str(n)] = output.shape
                base_conv = n.args[0]
                prev_conv = base_conv.args[0]
                base_conv_str = str(base_conv)
                prev_conv_str = str(prev_conv)
                if n.op == 'output':
                    self.lambda_output[base_conv_str] = 1.0
                else:
                    lambda_output = np.percentile(output.detach().reshape(-1), 99.9)
                    if lambda_output == 0:
                        lambda_output = 1.0
                    self.lambda_output[base_conv_str] = lambda_output

                if base_conv_str == 'conv1':
                    self.lambda_input[base_conv_str] = 1.0
                elif 'pooling' in prev_conv_str:
                    self.lambda_input[base_conv_str] = self.lambda_output[str(prev_conv.args[0].args[0])]
                elif 'cat' in prev_conv_str:
                    dim = prev_conv.kwargs['dim']
                    lambda_input = {}
                    for idx, arg in enumerate(prev_conv.args[0]):
                        dim_shape = self.shape_dict[str(arg)][dim]
                        lambda_input_idx = self.lambda_output[str(arg.args[0])]
                        lambda_input[str(idx)] = {'dim': dim, 'shape': dim_shape, 'scale': lambda_input_idx}
                    self.lambda_input[base_conv_str] = lambda_input
                else:
                    self.lambda_input[base_conv_str] = self.lambda_output[str(prev_conv.args[0])]

        return output

class torch_kernel(object):
    def __init__(self, path, method='connection_wise', scale_method='robust') -> None:
        self.lambda_input = {}
        self.lambda_output = {}
        self.method = method
        self.scale_method = scale_method

        prefix_path = os.path.join(path, self.scale_method)
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)

        if self.method == 'connection_wise':
            self.input_path = os.path.join(path, self.scale_method, "lambda_input_connection_wise.npy")
            self.output_path = os.path.join(path, self.scale_method, "lambda_output_connection_wise.npy")

        if os.path.exists(self.input_path) and os.path.exists(self.output_path):
            self.lambda_input = np.load(self.input_path, allow_pickle=True).item()
            self.lambda_output = np.load(self.output_path, allow_pickle=True).item()

    def scale_weight(self, model):
        for name, module in model.named_modules():
            if name in self.lambda_input.keys():
                lambda_input = self.lambda_input[name]
                lambda_output = self.lambda_output[name]
                if isinstance(module, nn.Conv2d):
                    # print(module.weight.data.shape)
                    if isinstance(lambda_input, dict):
                        for idx, content in lambda_input.items():
                            assert content['dim'] == 1
                            start = int(idx) * content['shape']
                            end = start + content['shape'] - 1
                            scale_factor = content['scale']
                            # print(start, end)
                            module.weight.data[:, start:end, :, :] = module.weight.data[:, start:end, :, :] *  scale_factor
                    else:
                        module.weight.data = module.weight.data * lambda_input

                    module.weight.data = module.weight.data / lambda_output

                elif isinstance(module, nn.ConvTranspose2d):
                    if isinstance(lambda_input, dict):
                        for idx, content in lambda_input.items():
                            assert content['dim'] == 1
                            start = int(idx) * content['shape']
                            end = start + content['shape'] - 1
                            scale_factor = content['scale']
                            module.weight.data[start:end, :, :, :] = module.weight.data[start:end, :, :, :] *  scale_factor
                    else:
                        module.weight.data = module.weight.data * lambda_input

                    module.weight.data = module.weight.data / lambda_output
                else:
                    pass

                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data / lambda_output

        return model

    def weight_normalization(self, model: nn.Module, norm_data: torch.Tensor()) -> nn.Module:
        if self.lambda_input and self.lambda_output:
            scale_model = self.scale_weight(model)
        else:
            with torch.no_grad():
                kernel_trace = TorchKernelInterpreter(model)
                kernel_trace.run(norm_data, "original")
                self.lambda_input  = kernel_trace.lambda_input
                self.lambda_output = kernel_trace.lambda_output
                scale_model = self.scale_weight(model)
                np.save(self.input_path, self.lambda_input)
                np.save(self.output_path, self.lambda_output)
        return scale_model


