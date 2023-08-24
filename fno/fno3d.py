import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from ..utils.utilities3 import *
from timeit import default_timer


################################################################
# 3d fourier layers
################################################################

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

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        #print("HEHEHE: ", x_ft[:, :, :self.modes1, :self.modes2, :self.modes3].shape)
        #print("WEIGHTS1 SHAPE: ", self.weights1.shape)
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


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3,  days_in, days_out, gfs_in, in_channels_named, width=16):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.days_out = days_out
        self.width = width
        self.padding = 6  # pad the domain if input is non-periodic


        print("DAYS_OUT: ", days_out)
        print("DAYS_IN: ", days_in)
        print("DAYS_GFS: ", gfs_in)
        #print("CHANNELS_IN: ", channels_in)


        channel_types = int((in_channels_named - 1)/(gfs_in + 1) + 1)
        #self.p = nn.Linear(int(channels_in/(days_in+gfs_in))+3,
        self.p = nn.Linear(channel_types+3,
                           self.width)  # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        #self.fc1 = nn.Linear(self.width, 128)
        self.register_parameter('fc1', torch.nn.Parameter(torch.randn(self.width, 128)))
        #self.fc2 = nn.Linear(128, 1)
        self.register_parameter('fc2', torch.nn.Parameter(torch.randn(128, 1)))
        self.register_parameter('fc3', torch.nn.Parameter(torch.randn(days_in+gfs_in, days_out)))

        #self.q = MLP(self.width, 1, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):
        #print()
        #x = torch.squeeze(x, axis=2)
        #print("X SHAPE: ", x.shape)
        x = x.permute(0, 3, 4, 1, 2)
        #print("X SHAPE AFTER PERMUTE: ", x.shape)
        grid = self.get_grid(x.shape, x.device)
        #print("GRID SHAPE: ", grid.shape)
        x = torch.cat((x, grid), dim=-1)
        #print("X SHAPE BEFORE P: ", x.shape)

        x = self.p(x)
        #print("X AFTER P SHAPE: ", x.shape)
        x = x.permute(0, 4, 1, 2, 3)
        #x = x.permute(0, 4, 2, 3, 1)
        #print("PADDING: ", self.padding)
        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic
        #print("X AFTER PADDING SHAPE: ", x.shape)
        #x = x.permute(0, 4, 2, 3, 1)
        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        #print("X SHAPE ALMOST FINAL: ", x.shape)
        #x = self.fc1(x)
        x = torch.einsum("bixyz,io->boxyz", x, self.fc1)
        x = torch.einsum("bixyz,io->boxyz", x, self.fc2)
        #x = self.fc2(x)

        x = x.permute(0, 4, 1, 2, 3)
        #print("X SHAPE ALMOST-ALMOST FINAL: ", x.shape)
        #print("FC3 SHAPE: ", self.fc3.shape)

        #x = self.fc1(x)
        #x = torch.einsum("bixyz,io->boxyz", x, self.fc1)
        #x = self.fc2(x)
        x = torch.einsum("bzxyi,zw->bwxyi", x, self.fc3)

        #x = self.q(x)
        #x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        print("X FINAL OF FORWARD: ", x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
