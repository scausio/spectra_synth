import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import random
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class DsFFNN_1():
    def __init__(self,ds, test_size=0.2):

        self.test_size=test_size
        # Prepare the dataset
        self.x_in = np.vstack([ds.SWH, ds.WPP, ds.MWP, ds.MWD,
                               ds.PWD, ds.DPT]).T

        self.y_in = ds.EF
        print(f"SWH valid shape:{ds.SWH.shape}")
        self.split()
        #self.scale()
        self.normalize()

    def split(self):
        print ('Splitting train and test')
        # Split the dataset
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.x_in, self.y_in, test_size=self.test_size, random_state=42)

    def scale(self):
        print ('Standardize the data')
        # Standardize the features
        scaler_X = StandardScaler().fit(self.X_train)
        scaler_Y = StandardScaler().fit(self.Y_train)

        self.scaler_X=scaler_X
        self.scaler_Y=scaler_Y

        X_train_scaled = scaler_X.transform(self.X_train)
        X_test_scaled = scaler_X.transform(self.X_test)

        self.Y_train = scaler_Y.transform(self.Y_train)
        self.Y_test = scaler_Y.transform(self.Y_test)

        self.X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
        self.X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
        self.Y_train = torch.tensor(self.Y_train, dtype=torch.float32)
        self.train_dataset = TensorDataset(self.X_train_scaled, self.Y_train)

    def normalize(self):
        print ('Normalize the data')
        scaler_X = MinMaxScaler(feature_range=(0.2, 0.8)).fit(self.X_train)
        scaler_Y = MinMaxScaler(feature_range=(0.2, 0.8)).fit(self.Y_train)

        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

        X_train_scaled = scaler_X.transform(self.X_train)
        X_test_scaled = scaler_X.transform(self.X_test)
        self.Y_train = scaler_Y.transform(self.Y_train)
        self.Y_test = scaler_Y.transform(self.Y_test)

        self.X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
        self.X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
        self.Y_train = torch.tensor(self.Y_train, dtype=torch.float32)
        self.train_dataset = TensorDataset(self.X_train_scaled, self.Y_train)

class Ds_Conv():

    def __init__(self, ds, reshape_size,wind=False,coords=False,visualize=False,
                 ):
        # Prepare the dataset
        self.getMP(ds,wind,coords,visualize)
        self.y_in = ds.EF
        self.reshape_size = reshape_size

        self.split()
        self.normalize()
    def getX_Yam(self,ds):
        self.x_in = np.vstack([ds.SWH, ds.WPP,
                               ds.MWD, ]).T

    def getMP(self,ds,wind,coords,visualize):

        print('coords',coords)
        print ('wind',wind)
        print ('visualize',visualize)

        if visualize:
            self.x_in = np.vstack([ds.SWH, ds.WPP, ds.MWD,
                                   ds.SWH_SW1, ds.WPP_SW1, ds.MWP_SW1,
                                   ds.SWH_SW2, ds.WPP_SW2, ds.MWP_SW2]).T
        else:
            self.x_in = np.vstack([ds.SWH_WW, ds.WPP_WW, ds.MWP_WW,
                                   ds.SWH_SW1, ds.WPP_SW1, ds.MWP_SW1,
                                   ds.SWH_SW2, ds.WPP_SW2, ds.MWP_SW2, ds.DPT]).T

        if coords:
            self.x_in = np.hstack([self.x_in, np.expand_dims(ds.LON, 1), np.expand_dims(ds.LAT, 1)])
        if wind:
            self.x_in = np.hstack([self.x_in, np.expand_dims(ds.u10, 1), np.expand_dims(ds.v10, 1)])

        print ('x_in shape:',self.x_in.shape)


    def scale(self):
        print ('Standardize the data')
        # Standardize the features
        scaler_X = StandardScaler().fit(self.X_train)
        scaler_Y = StandardScaler().fit(self.Y_train)

        self.scaler_X=scaler_X
        self.scaler_Y=scaler_Y

        X_train_scaled = scaler_X.transform(self.X_train)
        X_test_scaled = scaler_X.transform(self.X_test)

        self.Y_train = scaler_Y.transform(self.Y_train).reshape(len(self.Y_train), self.reshape_size[0], self.reshape_size[1])
        self.Y_test = scaler_Y.transform(self.Y_test).reshape(len(self.Y_test), self.reshape_size[0], self.reshape_size[1])

        self.X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
        self.X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
        self.Y_train = torch.tensor(self.Y_train, dtype=torch.float32)
        self.train_dataset = TensorDataset(self.X_train_scaled, self.Y_train)
        self.val_dataset = TensorDataset(self.X_test_scaled, self.Y_test)

    def normalize(self):
        print ('Normalize the data')
        scaler_X = MinMaxScaler(feature_range=(0.2, 0.8)).fit(self.X_train)
        scaler_Y = MinMaxScaler(feature_range=(0.2, 0.8)).fit(self.Y_train)

        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

        # Keep raw (unscaled) test data for inference
        self.X_test_raw = self.X_test.copy()
        self.Y_test_raw = self.Y_test.copy()

        X_train_scaled = scaler_X.transform(self.X_train)
        X_test_scaled = scaler_X.transform(self.X_test)
        Y_train_scaled = scaler_Y.transform(self.Y_train).reshape(len(self.Y_train), self.reshape_size[0], self.reshape_size[1])
        Y_test_scaled = scaler_Y.transform(self.Y_test).reshape(len(self.Y_test), self.reshape_size[0], self.reshape_size[1])

        self.X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
        self.X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
        self.Y_train = torch.tensor(Y_train_scaled, dtype=torch.float32)
        self.Y_test = torch.tensor(Y_test_scaled, dtype=torch.float32)
        self.train_dataset = TensorDataset(self.X_train_scaled, self.Y_train)
        self.val_dataset = TensorDataset(self.X_test_scaled, self.Y_test)

