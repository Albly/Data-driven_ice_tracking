# SOURCE: https://github.com/gegewen/ufno

import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial



class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        #print("X_FT SHAPE: ", x_ft.shape)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        #print("INPUT SHAPE: ", x_ft[:, :, :self.modes1, :self.modes2, :self.modes3].shape)
        #print("WEIGHTS SHAPE: ", self.weights1.shape)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                 dropout_rate=dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2,
                               dropout_rate=dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1,
                                 dropout_rate=dropout_rate)

        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels * 2, output_channels)
        self.deconv0 = self.deconv(input_channels * 2, output_channels)

        self.output_layer = self.output(input_channels * 2, output_channels,
                                        kernel_size=kernel_size, stride=1, dropout_rate=dropout_rate)

    def forward(self, x):

        def cat_with_crop(x1, x2):
            shape = x1.shape
            x2 = x2[:, :, :shape[2], :shape[3], :shape[4]]
            x = torch.cat([x2, x1], dim=1)
            return x

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_deconv2 = self.deconv2(out_conv3)

        #print("OUT CONV 1 SHAPE: ", out_conv1.shape)
        #print("OUT CONV 2 SHAPE: ", out_conv2.shape)
        #print("OUT CONV 3 SHAPE: ", out_conv3.shape)
        #print("OUT DECONV 2 SHAPE: ", out_deconv2.shape)

        concat2 = cat_with_crop(out_conv2, out_deconv2)
        #print("HEHEHE")
        out_deconv1 = self.deconv1(concat2)
        #print("OUT DECONV 1 SHAPE: ", out_deconv1.shape)
        concat1 = cat_with_crop(out_conv1, out_deconv1)
        out_deconv0 = self.deconv0(concat1)
        #print("OUT DECONV 0 SHAPE: ", out_deconv0.shape)
        concat0 = cat_with_crop(x, out_deconv0)
        #print("OUT CONCAT 0 SHAPE: ", concat0.shape)
        out = self.output_layer(concat0)
        #print("OUT: ", out.shape)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv3d(in_planes, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)


class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, days_in, days_out, width):
        super(SimpleBlock3d, self).__init__()
        """
        U-FNO contains 3 Fourier layers and 3 U-Fourier layers.

        input shape: (batchsize, x=200, y=96, t=24, c=12)
        output shape: (batchsize, x=200, y=96, t=24, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.days_out = days_out
        self.fc0 = nn.Linear(1, self.width)  # 1 channel

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv5 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)
        self.unet3 = UNet(self.width, self.width, 3, 0)
        self.unet4 = UNet(self.width, self.width, 3, 0)
        self.unet5 = UNet(self.width, self.width, 3, 0)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        # self.fc3 = nn.Linear(days_in + 8, days_out + 8)
        self.register_parameter('fc3', torch.nn.Parameter(torch.randn(days_in + 8, days_out + 8)))

    def forward(self, x):

        batchsize = x.shape[0]
        #print("X SHAPE ORIG: ", x.shape)
        #size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]
        size_z, size_x, size_y  = x.shape[1], x.shape[2], x.shape[3]
        #print("X SHAPE BEFORE FC0: ", x.shape)
        x = self.fc0(x)
        #x = x.permute(0, 4, 1, 2, 3)
        x = x.permute(0, 4, 2, 3, 1)
        #print("X SHAPE BEFORE CONV0: ", x.shape)
        x1 = self.conv0(x)
        #print("X SHAPE AFTER CONV0: ", x.shape)
        x2 = self.w0(x.reshape(batchsize, self.width, -1)).reshape(batchsize, self.width, size_x, size_y, size_z)
        #print("X1 SHAPE: ", x1.shape)
        #print("X2 SHAPE: ", x2.shape)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.reshape(batchsize, self.width, -1)).reshape(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.reshape(batchsize, self.width, -1)).reshape(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.reshape(batchsize, self.width, -1)).reshape(batchsize, self.width, size_x, size_y, size_z)
        #print("X BEFORE UNET SHAPE: ", x.shape)
        x3 = self.unet3(x)
        x = x1 + x2 + x3
        x = F.relu(x)

        x1 = self.conv4(x)
        x2 = self.w4(x.reshape(batchsize, self.width, -1)).reshape(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet4(x)
        x = x1 + x2 + x3
        x = F.relu(x)


        x1 = self.conv5(x)
        x2 = self.w5(x.reshape(batchsize, self.width, -1)).reshape(batchsize, self.width, size_x, size_y, size_z)
        #print("HEHEHEHEHE: ", x2.shape)
        x3 = self.unet5(x)
        x = x1 + x2 + x3
        x = F.relu(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        #print("X FINAL SHAPE: ", x.shape)

        x = torch.einsum("bxyzi,zw->bxywi", x, self.fc3)

        #print("X VERY FINAL SHAPE: ", x.shape)

        return x


class UFNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, days_in, days_out, width=8):
        super(UFNO3d, self).__init__()

        """
        A wrapper function
        """
        self.conv1 = SimpleBlock3d(modes1, modes2, modes3, days_in=days_in, days_out=days_out, width=width)

    def forward(self, x):
        #print("X SHAPE VERY VERY BEGINNING: ", x.shape)
        x = x.permute(0, 1, 3, 4, 2)
        batchsize = x.shape[0]
        size_z, size_x, size_y = x.shape[1], x.shape[2], x.shape[3]
        #print("AAAAAAAAAAAAAAAAAAAAAAAA X SHAPE: ", x.shape)
        x = F.pad(F.pad(x, (0, 0, 0, 8, 0, 8), "replicate"), (0, 0, 0, 0, 0, 0, 0, 8), 'constant', 0)
        x = self.conv1(x)
        x = x.view(batchsize, size_x + 8, size_y + 8, self.conv1.days_out + 8, 1)[..., :-8, :-8, :-8, :]
        #print("X SHAPE FINAL UFNO: ", x.shape)
        x = x.permute(0, 3, 4, 1, 2)
        #print("X AFTER PERMUTE FINAL SHAPE: ", x.shape)
        return x #x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c