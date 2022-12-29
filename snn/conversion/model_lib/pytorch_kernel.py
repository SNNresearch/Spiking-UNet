import torch 
import torch.nn as nn
import numpy as np
import os
class Link(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        return x

def absorb_to_parameter(BN_layer: nn.BatchNorm2d, prev_parameter_layer):

    BN_weight = BN_layer.weight.data
    BN_mu = BN_layer.running_mean.data
    BN_std = torch.sqrt(BN_layer.running_var.data + BN_layer.eps)
    prev_weight = prev_parameter_layer.weight.data
    
    if prev_parameter_layer.bias is not None:
        prev_bias = prev_parameter_layer.bias.data
    else:
        prev_bias = 0.0

    prev_weight = prev_weight * BN_weight.view(-1, 1, 1, 1) / BN_std.view(-1, 1, 1, 1)

    if BN_layer.affine:
        BN_bias = BN_layer.bias.data
        prev_bias = BN_weight * (prev_bias - BN_mu) / BN_std + BN_bias
    else:
        prev_bias = BN_weight * (prev_bias - BN_mu) / BN_std

class torch_kernel(object):
    def __init__(self, path, method='layer_wise', scale_method='robust') -> None:
        self.hook_list = []
        self.lambda_input = {}
        self.lambda_output = {}
        self.method = method
        self.scale_method = scale_method

        prefix_path = os.path.join(path, self.scale_method)
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)

        self.input_path = os.path.join(path, self.scale_method, "lambda_input_layer_wise.npy")
        self.output_path = os.path.join(path, self.scale_method, "lambda_output_layer_wise.npy")

        self.record_path = os.path.join(path, 'name_record.npy')

        if os.path.exists(self.input_path) and os.path.exists(self.output_path) and os.path.exists(self.record_path):
            self.lambda_input = np.load(self.input_path, allow_pickle=True).item()
            self.lambda_output = np.load(self.output_path, allow_pickle=True).item()
            self.name_record = np.load(self.record_path, allow_pickle=True).item()

    def transform(self, modules: nn.Module) -> nn.Module:
        for name, module in modules._modules.items():
            # process Sequential and BasicBlock model
            if module._modules:
                modules._modules[name] = self.transform(module)
            else:
                # Absorb Batch Norm parameters
                if isinstance(module, nn.BatchNorm2d):
                    try:
                        absorb_to_parameter(module, prev_parameter_layer)
                        new_layer = Link()
                        modules._modules[name] = new_layer
                    except:
                        print("Previous BN layer doesn't have parameter to absorb")
                else:
                    if hasattr(module, 'weight') and module.weight is not None:
                        prev_parameter_layer = module   

        return modules

    def layerwise_name_hook(self, layer_name: str, scale_method: str): 
        def hook(module, input, output): 
            if module.__class__.__name__ != 'ReLU' :
                input_activations = input[0].data.cpu().numpy().reshape(-1)
                if scale_method == 'robust':
                    input_scale_factor = np.percentile(input_activations, 99.9)
                else:
                    input_scale_factor = np.max(input_activations)
                del input_activations

                if input_scale_factor == 0.0:
                    self.lambda_input[layer_name] = 1.0
                else:
                    self.lambda_input[layer_name] = input_scale_factor
               

            if module.__class__.__name__ != 'Softmax2d':
                output_activations = output.data.cpu().numpy().reshape(-1)
                if scale_method == 'robust':
                    output_scale_factor = np.percentile(output_activations, 99.9)
                else:
                    output_scale_factor = np.max(output_activations)

                if output_scale_factor == 0.0:
                    self.lambda_output[layer_name] = 1.0
                else:
                    self.lambda_output[layer_name] = output_scale_factor
                del output_activations
            else:
                self.lambda_output[layer_name] = 1.0

        return hook
    
    def hook_state(self, modules, prefix='', prev_para_module='', idx=0, scale_method='robust'):
        for name, module in modules._modules.items():
            module_class = module.__class__.__name__
            if module._modules:
                if prefix:
                    submodule_prefix = prefix + '_' + name 
                else:
                    submodule_prefix = name
                input_name, output_name = self.hook_state(module, submodule_prefix, prev_para_module, idx, scale_method)
            else:
                if prefix:
                    module_name = prefix + '_' + name
                else:
                    module_name = name

                if idx == 0:
                    input_name = module_name
                    idx = 1
                
                # Replace the Conv 99.9 value with ReLU 99.9 value
                if hasattr(module, 'weight') and module.weight is not None:
                    prev_para_module = module_name
                elif module_class in ['ReLU', 'Softmax2d']:
                    module_name = prev_para_module

                self.hook_list.append(module.register_forward_hook(self.layerwise_name_hook(module_name, scale_method)))

        output_name = module_name
        return input_name, output_name

    def scale_weight(self, modules, input_name, output_name, prefix = '') -> nn.Module:
        for name, module in modules._modules.items():
            module_class = module.__class__.__name__
            if module._modules:
                if prefix:
                    submodule_prefix = prefix + '_' + name
                else:
                    submodule_prefix = name
                modules._modules[name] = self.scale_weight(module, input_name, output_name, submodule_prefix)
            else:
                if prefix:
                        module_name = prefix + '_' + name
                else:
                    module_name = name
                
                if module_name == input_name:
                    self.lambda_input[module_name] = np.ones_like(self.lambda_input[module_name])
                if module_name == output_name:
                    self.lambda_output[module_name] = np.ones_like(self.lambda_input[module_name])

                if module_class in ['Conv2d', 'Linear', 'ConvTranspose2d']:
                    lambda_input = self.lambda_input[module_name]
                    lambda_output = self.lambda_output[module_name]
                    module.weight.data = module.weight.data * lambda_input / lambda_output

                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.data = module.bias.data / lambda_output
        return modules

    def weight_normalization(self, model: nn.Module, norm_data: torch.Tensor(), print_in_out=True) -> nn.Module:
        if self.lambda_input and self.lambda_output:
            input_name = self.name_record['input_name']
            output_name = self.name_record['output_name']
            scale_model = self.scale_weight(model, input_name, output_name)
        else:
            with torch.no_grad():
                input_name, output_name = self.hook_state(model, prefix='', prev_para_module='', idx=0, scale_method=self.scale_method)
                first_layer = model.conv1(norm_data)
                model.forward(first_layer, 'img')
                scale_model = self.scale_weight(model, input_name, output_name)
                for hook_handle in self.hook_list:
                    hook_handle.remove()

                self.name_record = {'input_name': input_name, 'output_name': output_name}
                np.save(self.record_path, self.name_record)
                np.save(self.input_path, self.lambda_input)
                np.save(self.output_path, self.lambda_output)

        if print_in_out:
            print('input_name:{}, output_name:{}'.format(input_name, output_name))

        return scale_model

