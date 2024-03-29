from torch.nn import Module
from torch import nn
import torch
import sys
sys.path.append("/home/lihb/Github/bita_new_version")
import snn.spiking_neuron as neuron

class Denoising_Model(Module):
    def __init__(self, color=True, stbp=False, reset_method='reduce_by_zero'):
        super(Denoising_Model, self).__init__()
        if color:
            self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1, bias = True)
        else:
            self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, padding = 1, bias = True)

        if stbp:
            self.relu1              = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu2              = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu3              = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu4              = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu5              = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu6              = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu7              = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu8              = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu9              = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu10             = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu_convtranpose1 = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu11             = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu12             = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu_convtranpose2 = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu13             = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu14             = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu_convtranpose3 = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu15             = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu16             = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu_convtranpose4 = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu17             = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu18             = neuron.IF_STBP(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
        else:
            self.relu1              = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu2              = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu3              = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu4              = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu5              = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu6              = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu7              = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu8              = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu9              = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu10             = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu_convtranpose1 = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu11             = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu12             = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu_convtranpose2 = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu13             = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu14             = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu_convtranpose3 = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu15             = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu16             = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu_convtranpose4 = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu17             = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)
            self.relu18             = neuron.IF_Neuron(v_threshold = 1.0,  v_reset = 0.0, reset_method = reset_method)


        self.conv2              = nn.Conv2d(in_channels          = 64,   out_channels = 64, kernel_size   = 3, padding = 1, bias    = True)
        self.pooling1           = nn.AvgPool2d(kernel_size       = 2,    padding      = 0)
        self.conv3              = nn.Conv2d(in_channels          = 64,   out_channels = 128, kernel_size  = 3, padding = 1, bias    = True)
        self.conv4              = nn.Conv2d(in_channels          = 256,  out_channels = 128, kernel_size  = 3, padding = 1, bias    = True)
        #                       self.conv4            = nn.Conv2d(in_channels          = 128,  out_channels = 128, kernel_size  = 3, padding                     = 1, bias    = True)
        self.pooling2           = nn.AvgPool2d(kernel_size       = 2,    padding      = 0)
        self.conv5              = nn.Conv2d(in_channels          = 128,  out_channels = 256, kernel_size  = 3, padding = 1, bias    = True)
        self.conv6              = nn.Conv2d(in_channels          = 512,  out_channels = 256, kernel_size  = 3, padding = 1, bias    = True)
        #                       self.conv6            = nn.Conv2d(in_channels          = 256,  out_channels = 256, kernel_size  = 3, padding                     = 1, bias    = True)
        self.pooling3           = nn.AvgPool2d(kernel_size       = 2,    padding      = 0)
        self.conv7              = nn.Conv2d(in_channels          = 256,  out_channels = 512, kernel_size  = 3, padding = 1, bias    = True)
        self.conv8              = nn.Conv2d(in_channels          = 1024,  out_channels = 512, kernel_size  = 3, padding= 1, bias    = True)
        #                       self.conv8            = nn.Conv2d(in_channels          = 512,  out_channels = 512, kernel_size  = 3, padding                     = 1, bias    = True)
        self.pooling4           = nn.AvgPool2d(kernel_size       = 2,    padding      = 0)
        self.conv9              = nn.Conv2d(in_channels          = 512,  out_channels = 1024, kernel_size = 3, padding = 1, bias    = True)
        self.conv10             = nn.Conv2d(in_channels          = 2048, out_channels = 1024, kernel_size = 3, padding = 1, bias    = True)
        #                       self.conv10           = nn.Conv2d(in_channels          = 1024, out_channels = 1024, kernel_size = 3, padding                     = 1, bias    = True)
        self.conv11             = nn.Conv2d(in_channels          = 1024, out_channels = 512, kernel_size  = 3, padding     = 1, bias    = True)
        self.conv12             = nn.Conv2d(in_channels          = 512,  out_channels = 512, kernel_size  = 3, padding     = 1, bias    = True)
        self.conv13             = nn.Conv2d(in_channels          = 512,  out_channels = 256, kernel_size  = 3, padding     = 1, bias    = True)
        self.conv14             = nn.Conv2d(in_channels          = 256,  out_channels = 256, kernel_size  = 3, padding     = 1, bias    = True)
        self.conv15             = nn.Conv2d(in_channels          = 256,  out_channels = 128, kernel_size  = 3, padding     = 1, bias    = True)
        self.conv16             = nn.Conv2d(in_channels          = 128,  out_channels = 128, kernel_size  = 3, padding     = 1, bias    = True)
        self.conv17             = nn.Conv2d(in_channels          = 128,  out_channels = 64, kernel_size   = 3, padding     = 1, bias    = True)
        self.conv18             = nn.Conv2d(in_channels          = 64,   out_channels = 64, kernel_size   = 3, padding     = 1, bias    = True)

        self.convtranspose1     = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, stride       = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        self.convtranspose2     = nn.ConvTranspose2d(in_channels = 512,  out_channels = 256, stride       = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        self.convtranspose3     = nn.ConvTranspose2d(in_channels = 256,  out_channels = 128, stride       = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        self.convtranspose4     = nn.ConvTranspose2d(in_channels = 128,  out_channels = 64, stride        = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)

        if color:
            self.conv19 = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 3, padding = 1, bias = True)
        else:
            self.conv19 = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1, bias = True)

        self.conv_d1 = nn.Conv2d(in_channels = 64, out_channels  = 128, kernel_size  = 2, stride = 2, bias = True)
        self.conv_d2 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size  = 2, stride = 2, bias = True)
        self.conv_d3 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size  = 2, stride = 2, bias = True)
        self.conv_d4 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 2, stride = 2, bias = True)
    
    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.relu1(x_1)

        x_1 = self.conv2(x_1)
        x_1 = self.relu2(x_1)
        x_d1 = self.conv_d1(x_1)
        down_x1 = self.pooling1(x_1)
        x_2 = self.conv3(down_x1)
        x_2 = self.relu3(x_2)
        x_2 = torch.cat((x_d1, x_2), dim = 1)
        x_2 = self.conv4(x_2)
        x_2 = self.relu4(x_2)
        x_d2 = self.conv_d2(x_2)
        down_x2 = self.pooling2(x_2)
        x_3 = self.conv5(down_x2)
        x_3 = self.relu5(x_3)
        x_3 = torch.cat((x_d2, x_3), dim = 1)
        x_3 = self.conv6(x_3)
        x_3 = self.relu6(x_3)
        x_d3 = self.conv_d3(x_3)
        down_x3 = self.pooling3(x_3)
        x_4 = self.conv7(down_x3)
        x_4 = self.relu7(x_4)
        x_4 = torch.cat((x_d3, x_4), dim = 1)
        x_4 = self.conv8(x_4)
        x_4 = self.relu8(x_4)
        x_d4 = self.conv_d4(x_4)
        down_x4 = self.pooling4(x_4)
        x_5 = self.conv9(down_x4)
        x_5 = self.relu9(x_5)
        x_5 = torch.cat((x_d4, x_5), dim = 1)
        x_5 = self.conv10(x_5)
        x_5 = self.relu10(x_5)
        up1 = self.convtranspose1(x_5)
        up1 = self.relu_convtranpose1(up1)
        concat1 = torch.cat((x_4, up1), dim = 1)
        x_6 = self.conv11(concat1)
        x_6 = self.relu11(x_6)
        x_6 = self.conv12(x_6)
        x_6 = self.relu12(x_6)
        up2 = self.convtranspose2(x_6)
        up2 = self.relu_convtranpose2(up2)
        concat2 = torch.cat((x_3, up2), dim = 1)
        x_7 = self.conv13(concat2)
        x_7 = self.relu13(x_7)
        x_7 = self.conv14(x_7)
        x_7 = self.relu14(x_7)
        up3 = self.convtranspose3(x_7)
        up3 = self.relu_convtranpose3(up3)
        concat3 = torch.cat((x_2, up3), dim = 1)
        x_8 = self.conv15(concat3)
        x_8 = self.relu15(x_8)
        x_8 = self.conv16(x_8)
        x_8 = self.relu16(x_8)
        up4 = self.convtranspose4(x_8)
        up4 = self.relu_convtranpose4(up4)
        concat4 = torch.cat((x_1, up4), dim = 1)
        x_9 = self.conv17(concat4)
        x_9 = self.relu17(x_9)
        x_9 = self.conv18(x_9)
        x_9 = self.relu18(x_9)
        output = self.conv19(x_9)
        
        return output
