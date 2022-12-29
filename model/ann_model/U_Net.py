from torch.nn import Module
from torch import nn
import torch

class Denoising_Model(Module):
    def __init__(self, color = True):
        super(Denoising_Model, self).__init__()

        if color:
            self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1, bias = True)
        else:
            self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, padding = 1, bias = True)

        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = True)
        self.relu2 = nn.ReLU()
        self.pooling1 = nn.AvgPool2d(kernel_size = 2, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, bias = True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True)
        self.relu4 = nn.ReLU()
        self.pooling2 = nn.AvgPool2d(kernel_size = 2, padding = 0)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1, bias = True)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1, bias = True)
        self.relu6 = nn.ReLU()
        self.pooling3 = nn.AvgPool2d(kernel_size = 2, padding = 0)
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1, bias = True)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1, bias = True)
        self.relu8 = nn.ReLU()
        self.pooling4 = nn.AvgPool2d(kernel_size = 2, padding = 0)
        self.conv9 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, padding = 1, bias = True)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, padding = 1, bias = True)
        self.relu10 = nn.ReLU()
        self.convtranspose1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, stride = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        self.relu_convtranpose1 = nn.ReLU()
        self.conv11 = nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 3, padding = 1, bias = True)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1, bias = True)
        self.relu12 = nn.ReLU()
        self.convtranspose2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, stride = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        self.relu_convtranpose2 = nn.ReLU()
        self.conv13 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, padding = 1, bias = True)
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1, bias = True)
        self.relu14 = nn.ReLU()
        self.convtranspose3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, stride = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        self.relu_convtranpose3 = nn.ReLU()
        self.conv15 = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, padding = 1, bias = True)
        self.relu15 = nn.ReLU()
        self.conv16 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True)
        self.relu16 = nn.ReLU()
        self.convtranspose4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, stride = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        self.relu_convtranpose4 = nn.ReLU()
        self.conv17 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True)
        self.relu17 = nn.ReLU()
        self.conv18 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, bias = True)
        self.relu18 = nn.ReLU()
        if color:
            self.conv19 = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 3, padding = 1, bias = True)
        else:
            self.conv19 = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, padding = 1, bias = True)

    def forward(self, x, input_type):
        
        if input_type == 'original':
            x_1 = self.conv1(x)
            x_1 = self.relu1(x_1)
        elif input_type == 'img':
            # x_1 = self.conv1(x)
            x_1 = self.relu1(x)
        else:
            x_1 = x

        x_1 = self.conv2(x_1)
        x_1 = self.relu2(x_1)
        down_x1 = self.pooling1(x_1)
        x_2 = self.conv3(down_x1)
        x_2 = self.relu3(x_2)
        x_2 = self.conv4(x_2)
        x_2 = self.relu4(x_2)
        down_x2 = self.pooling2(x_2)
        x_3 = self.conv5(down_x2)
        x_3 = self.relu5(x_3)
        x_3 = self.conv6(x_3)
        x_3 = self.relu6(x_3)
        down_x3 = self.pooling3(x_3)
        x_4 = self.conv7(down_x3)
        x_4 = self.relu7(x_4)
        x_4 = self.conv8(x_4)
        x_4 = self.relu8(x_4)
        down_x4 = self.pooling4(x_4)
        x_5 = self.conv9(down_x4)
        x_5 = self.relu9(x_5)
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

class parallel_v4(Module):
    def __init__(self, model) -> None:
        super(parallel_v4, self).__init__()
        self.encoder_1 = nn.Sequential(
            model.conv1,
            model.relu1,
            model.conv2,
            model.relu2,
        ).to('cuda:0')
        self.encoder_2 = nn.Sequential(
            model.pooling1,
            model.conv3,
            model.relu3,
            model.conv4,
            model.relu4
        ).to('cuda:0')
        self.encoder_3 = nn.Sequential(
            model.pooling2,
            model.conv5,
            model.relu5,
            model.conv6,
            model.relu6
        ).to('cuda:0')
        self.encoder_4 = nn.Sequential(
            model.pooling3,
            model.conv7,
            model.relu7,
            model.conv8,
            model.relu8
        ).to('cuda:0')
        self.down_up_1 = nn.Sequential(
            model.pooling4,
            model.conv9,
            model.relu9,
            model.conv10,
            model.relu10,
            model.convtranspose1,
            model.relu_convtranpose1,
        ).to('cuda:1')
        self.down_up_2 = nn.Sequential(
            model.conv11,
            model.relu11,
            model.conv12,
            model.relu12,
            model.convtranspose2,
            model.relu_convtranpose2
        ).to('cuda:1')
        self.down_up_3 = nn.Sequential(
            model.conv13,
            model.relu13,
            model.conv14,
            model.relu14,
            model.convtranspose3,
            model.relu_convtranpose3
        ).to('cuda:1')
        self.down_up_4 = nn.Sequential(
            model.conv15,
            model.relu15,
            model.conv16,
            model.relu16,
            model.convtranspose4,
            model.relu_convtranpose4
        ).to('cuda:1')
        self.output = nn.Sequential(
            model.conv17,
            model.relu17,
            model.conv18,
            model.relu18,
            model.conv19
        ).to('cuda:1')

    def forward(self, x):
        x1 = self.encoder_1(x.to('cuda:0'))
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.encoder_4(x3)
        # x2 = self.encoder_2(x1.to('cuda:0'))
        # x3 = self.encoder_3(x2.to('cuda:0'))
        # x4 = self.encoder_4(x3.to('cuda:0'))
        x5 = self.down_up_1(x4.to('cuda:1'))
        concat1 = torch.cat((x4.to('cuda:1'), x5.to('cuda:1')), dim = 1) 
        x6 = self.down_up_2(concat1)
        concat2 = torch.cat((x3.to('cuda:1'), x6.to('cuda:1')), dim = 1) 
        x7 = self.down_up_3(concat2)
        concat3 = torch.cat((x2.to('cuda:1'), x7.to('cuda:1')), dim = 1) 
        x8 = self.down_up_4(concat3)
        concat4 = torch.cat((x1.to('cuda:1'), x8.to('cuda:1')), dim = 1) 
        output = self.output(concat4)
        return output

class Segmentation_UNet(Module):
    def __init__(self, input_channel = 1, class_num = 2, fnum = 64):
        super(Segmentation_UNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = input_channel, out_channels = fnum, kernel_size = 3, padding = 1, bias = True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = fnum, out_channels = fnum, kernel_size = 3, padding = 1, bias = True)
        self.relu2 = nn.ReLU()
        self.pooling1 = nn.AvgPool2d(kernel_size = 2, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = fnum, out_channels = fnum * 2, kernel_size = 3, padding = 1, bias = True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels = fnum * 2, out_channels = fnum * 2, kernel_size = 3, padding = 1, bias = True)
        self.relu4 = nn.ReLU()
        self.pooling2 = nn.AvgPool2d(kernel_size = 2, padding = 0)
        self.conv5 = nn.Conv2d(in_channels = fnum * 2, out_channels = fnum * 4, kernel_size = 3, padding = 1, bias = True)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels = fnum * 4, out_channels = fnum * 4, kernel_size = 3, padding = 1, bias = True)
        self.relu6 = nn.ReLU()
        self.pooling3 = nn.AvgPool2d(kernel_size = 2, padding = 0)
        self.conv7 = nn.Conv2d(in_channels = fnum * 4, out_channels = fnum * 8, kernel_size = 3, padding = 1, bias = True)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels = fnum * 8, out_channels = fnum * 8, kernel_size = 3, padding = 1, bias = True)
        self.relu8 = nn.ReLU()
        self.pooling4 = nn.AvgPool2d(kernel_size = 2, padding = 0)
        self.conv9 = nn.Conv2d(in_channels = fnum * 8, out_channels = fnum * 16, kernel_size = 3, padding = 1, bias = True)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(in_channels = fnum * 16, out_channels = fnum * 16, kernel_size = 3, padding = 1, bias = True)
        self.relu10 = nn.ReLU()
        self.convtranspose1 = nn.ConvTranspose2d(in_channels = fnum * 16, out_channels = fnum * 8, stride = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        self.relu_convtranpose1 = nn.ReLU()
        self.conv11 = nn.Conv2d(in_channels = fnum * 16, out_channels = fnum * 8, kernel_size = 3, padding = 1, bias = True)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(in_channels = fnum * 8, out_channels = fnum * 8, kernel_size = 3, padding = 1, bias = True)
        self.relu12 = nn.ReLU()
        self.convtranspose2 = nn.ConvTranspose2d(in_channels = fnum * 8, out_channels = fnum * 4, stride = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        self.relu_convtranpose2 = nn.ReLU()
        self.conv13 = nn.Conv2d(in_channels = fnum * 8, out_channels = fnum * 4, kernel_size = 3, padding = 1, bias = True)
        self.relu13 = nn.ReLU()
        self.conv14 = nn.Conv2d(in_channels = fnum * 4, out_channels = fnum * 4, kernel_size = 3, padding = 1, bias = True)
        self.relu14 = nn.ReLU()
        self.convtranspose3 = nn.ConvTranspose2d(in_channels = fnum * 4, out_channels = fnum * 2, stride = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        self.relu_convtranpose3 = nn.ReLU()
        self.conv15 = nn.Conv2d(in_channels = fnum * 4, out_channels = fnum * 2, kernel_size = 3, padding = 1, bias = True)
        self.relu15 = nn.ReLU()
        self.conv16 = nn.Conv2d(in_channels = fnum * 2, out_channels = fnum * 2, kernel_size = 3, padding = 1, bias = True)
        self.relu16 = nn.ReLU()
        self.convtranspose4 = nn.ConvTranspose2d(in_channels = fnum * 2, out_channels = fnum, stride = 2, kernel_size = 3, padding = 1, output_padding = 1, bias = True)
        self.relu_convtranpose4 = nn.ReLU()
        self.conv17 = nn.Conv2d(in_channels = fnum * 2, out_channels = fnum, kernel_size = 3, padding = 1, bias = True)
        self.relu17 = nn.ReLU()
        self.conv18 = nn.Conv2d(in_channels = fnum, out_channels = fnum, kernel_size = 3, padding = 1, bias = True)
        self.relu18 = nn.ReLU()
        self.conv19 = nn.Conv2d(in_channels = fnum, out_channels = class_num, kernel_size = 1, bias = True)

    def forward(self, x, input_type):
        if input_type == 'original':
            x_1 = self.conv1(x)
            x_1 = self.relu1(x_1)
        elif input_type == 'img':
            # x_1 = self.conv1(x)
            x_1 = self.relu1(x)
        else:
            x_1 = x

        x_1 = self.conv2(x_1)
        x_1 = self.relu2(x_1)
        down_x1 = self.pooling1(x_1)
        x_2 = self.conv3(down_x1)
        x_2 = self.relu3(x_2)
        x_2 = self.conv4(x_2)
        x_2 = self.relu4(x_2)
        down_x2 = self.pooling2(x_2)
        x_3 = self.conv5(down_x2)
        x_3 = self.relu5(x_3)
        x_3 = self.conv6(x_3)
        x_3 = self.relu6(x_3)
        down_x3 = self.pooling3(x_3)
        x_4 = self.conv7(down_x3)
        x_4 = self.relu7(x_4)
        x_4 = self.conv8(x_4)
        x_4 = self.relu8(x_4)
        down_x4 = self.pooling4(x_4)
        x_5 = self.conv9(down_x4)
        x_5 = self.relu9(x_5)
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


