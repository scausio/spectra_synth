#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Conv1D, LSTM,\
#    ConvLSTM1D, LeakyReLU, Softmax
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import random
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False




class FFNN_0(nn.Module):
    def __init__(self, X_train_scaled, y_train):
        super(FFNN_0, self).__init__()
        input_dim = X_train_scaled.shape[1]
        output_dim = y_train.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class Conv_16(nn.Module):
    def __init__(self, X_train_scaled, reshape_size,scalerX,scalerY, units=256,resBlocks=4,simpleBlocks=4):
        super(Conv_16, self).__init__()
        input_dim = X_train_scaled.shape[1]
        #output_dim = y_train.shape[1]
        self.reshape_size = reshape_size
        self.fcIn = nn.Linear(input_dim, units)
        self.fc = nn.Linear(units, units)
        self.resBlocks=resBlocks
        self.simpleBlocks=simpleBlocks
        self.fc_to_conv = nn.Linear(units, self.reshape_size[0] * self.reshape_size[1])

        self.convIn = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.bnIn = nn.BatchNorm2d(16)
        self.convOut = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1)
        self.bnOut = nn.BatchNorm2d(1)
        self.leaky_relu = nn.LeakyReLU()
        self.scalerX = scalerX
        self.scalerY=scalerY

    def save_scaler(self, filepath):
        scalers = {'X': self.scalerX, 'Y': self.scalerY}
        joblib.dump(scalers, filepath)
    def forward(self, x):
        x = self.fcIn(x)  # Save the input for the skip connection
        x = self.fc_to_conv(x)
        x = self.leaky_relu(x)
        x = x.view(-1, 1, self.reshape_size[0], self.reshape_size[1])
        for i in range(9):
            x = self.convIn(x)
            x = self.leaky_relu(x)
            #x = self.maxpool(x)
            print(6, x.shape)
            x = self.convOut(x)
        return x
"""    def forward(self, x):

        x_shortcut = self.fcIn(x)  # Save the input for the skip connection
        # Residual blocks
        for i in range(self.resBlocks):
            x = self.fc(x_shortcut if i == 0 else x)  # First block uses x_shortcut
            x = self.leaky_relu(x + x_shortcut)  # Residual connection
        for i in range(self.simpleBlocks):
            x = self.fc(x)
            x = self.leaky_relu(x)
        #print('x', x.shape)
        x = self.fc_to_conv(x)
        x = self.leaky_relu(x)

        x = x.view(-1, 1, self.reshape_size[0], self.reshape_size[1])
        
        x = self.convIn(x)
        x = self.bnIn(x)
        #print('convin', x.shape)
        x = self.leaky_relu(x)
        x = self.convOut(x)
        #x = self.bnOut(x)
        x = self.leaky_relu(x)

        return x"""


class Conv_16_v2(nn.Module):
    def __init__(self, X_train_scaled, reshape_size, scalerX, scalerY, units=256,resBlocks=4, simpleBlocks=4):
        super(Conv_16_v2, self).__init__()
        input_dim = X_train_scaled.shape[1]
        self.reshape_size = reshape_size
        self.fcIn = nn.Linear(input_dim, units)

        #self.convIn = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        #self.convOut = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.res_blocks = ResidualSequential(units, resBlocks).to(device)

        simpleBlocks = [
                           module for _ in range(simpleBlocks)
                           for module in [nn.Linear(units, units), nn.LeakyReLU()]
                       ] + [
                           nn.Linear(units, self.reshape_size[0] * self.reshape_size[1]),
                           nn.LeakyReLU(),
                       ]
        self.simple_blocks = nn.Sequential(*simpleBlocks)

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),
        )

        self.leaky_relu = nn.LeakyReLU()
        self.scalerX = scalerX
        self.scalerY = scalerY

    def save_scaler(self, filepath):
        scalers = {'X': self.scalerX, 'Y': self.scalerY}
        joblib.dump(scalers, filepath)

    def forward(self, x):
        x = self.fcIn(x)
        x = self.res_blocks.forward(x)
        x=self.simple_blocks(x)
        x = x.view(-1, 1, self.reshape_size[0], self.reshape_size[1])
        x=self.conv_blocks(x)
        return x

class Conv_16_rev(nn.Module):
    def __init__(self, X_train_scaled, reshape_size, units=256):
        super(Conv_16_rev, self).__init__()
        input_dim = X_train_scaled.shape[1]
        # output_dim = y_train.shape[1]
        self.reshape_size = reshape_size
        self.fcIn = nn.Linear(input_dim, units)
        self.fc = nn.Linear(units, units)

        self.fc_to_conv = nn.Linear(units, self.reshape_size[0] * self.reshape_size[1])

        self.convIn = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)

        self.convOut = nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1)

        self.leaky_relu = nn.LeakyReLU()

        self.out = nn.Linear(units, self.reshape_size[0] * self.reshape_size[1])


    def forward(self, x):
        resBlocks = 4
        simpleBlocks = 4
        x_shortcut = self.fcIn(x)  # Save the input for the skip connection
        #print ('x_shortcut',x.shape)
        x = self.fc_to_conv(x_shortcut)
        x = x.view(-1, 1, self.reshape_size[0], self.reshape_size[1])
        x = self.convIn(x)
        #print('convin',x.shape)
        x = self.leaky_relu(x)
        x = self.convOut(x)
        #print('convout',x.shape)

        for i in range(resBlocks):
            if i == 0:
                x = self.fc(x_shortcut)
            else:
                x = self.fc(x)
            x = self.leaky_relu(x + x_shortcut)  # Residual
        for i in range(simpleBlocks):
            x = self.fc(x)
            x = self.leaky_relu(x)
        x=self.out(x)
        #print('out',x.shape)
        return x


class ResidualSequential(nn.Module):
    def __init__(self, units, blocks):
        super(ResidualSequential, self).__init__()
        self.my_children = [nn.Linear(units, units) for _ in range(blocks)]

    def to(self, *args, **kwargs):
        super(ResidualSequential, self).to(*args, **kwargs)
        [module.to(*args, **kwargs) for module in self.my_children]

    def forward(self, x):
        skip = x
        for module in self.my_children:
            x = module(x)
            x = nn.LeakyReLU()(x + skip)
        return x


class Conv16_fx(nn.Module):
    def __init__(self, X_train_scaled, reshape_size, scalerX,scalerY, units=256, resBlocks=4, simpleBlocks=4):
        super(Conv16_fx, self).__init__()
        input_dim = X_train_scaled.shape[1]
        self.reshape_size = reshape_size
        self.fcIn = nn.Linear(input_dim, units)
        self.res_blocks = ResidualSequential(units, resBlocks)
        simpleBlocks = [
                           module for _ in range(simpleBlocks)
                           for module in [nn.Linear(units, units), nn.LeakyReLU()]
                       ] + [
                           nn.Linear(units, self.reshape_size[0] * self.reshape_size[1]),
                           nn.LeakyReLU(),
                       ]
        self.simple_blocks = nn.Sequential(*simpleBlocks)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),
        )
        self.scalerX = scalerX
        self.scalerY = scalerY
    def save_scaler(self, filepath):
        scalers = {'X': self.scalerX, 'Y': self.scalerY}
        joblib.dump(scalers, filepath)
    def to(self, *args, **kwargs):
        super(Conv16_fx, self).to(*args, **kwargs)
        self.res_blocks.to(*args, **kwargs)

    def forward(self, x):
        x = self.fcIn(x)
        x = self.res_blocks.forward(x)
        x = self.simple_blocks(x)
        x = x.view(-1, 1, self.reshape_size[0], self.reshape_size[1])
        x = self.conv_blocks(x)
        print("Output of conv:", x)
        return x


# this is the ENCODER
class ReverseModel(nn.Module):
    def __init__(self, input_channels=1, image_height=32, image_width=24, num_outputs=8, ch1=4, ch2=6):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        assert image_height % 4 == 0 and image_width % 4 == 0, "width and height must be multiple of 4"
        flattened_size = ch2 * (image_height // 4) * (image_width // 4)
        logging.info(f"EMBEDDING SIZE: {flattened_size}")
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, ch1, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch1, ch2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(flattened_size, num_outputs)
        )

    def forward(self, x):
        img = x
        x = x.view(-1, 1, self.image_width, self.image_height)
        x = self.encoder(x)
        return x
