import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim

import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import pyarrow as pa
import time

'''
#READ FROM LOCAL
class CelestialDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.feature_columns = ['u', 'i', 'psfMag_z','petroRad_g', 'petroRad_r', 'petroRad_i', 'petroRad_z', 'expRad_u', 'expRad_g', 'expRad_r', 'expRad_i', 'expRad_z', 'ra', 'dec', 'l']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = torch.tensor(self.df.loc[idx, self.feature_columns].values, dtype=torch.float32)
        label = torch.tensor(self.df.loc[idx, 'type'], dtype=torch.float32)
        return features, label

csv_path = 'CUT.csv'
celestial_dataset = CelestialDataset(csv_path)

test_csv_path = 'testCUT.csv'
celestial_test_dataset = CelestialDataset(test_csv_path)
'''

#READ FROM HDFS
class CelestialDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.feature_columns = ['u', 'i', 'psfMag_z', 'petroRad_g', 'petroRad_r', 'petroRad_i', 'petroRad_z', 'expRad_u', 'expRad_g', 'expRad_r', 'expRad_i', 'expRad_z', 'ra', 'dec', 'l']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = torch.tensor(self.df.loc[idx, self.feature_columns].values, dtype=torch.float32)
        label = torch.tensor(self.df.loc[idx, 'type'], dtype=torch.float32)
        return features, label

def read_csv_from_hdfs(hdfs_path):
    hdfs = pa.hdfs.connect()
    with hdfs.open(hdfs_path, 'rb') as f:
        df = pd.read_csv(f)
    return df

# Use HDFS paths for your datasets
#csv_path = 'hdfs://okeanos-master:54310/CUT.csv'
csv_path = 'hdfs://okeanos-master:54310/processed_celestial_data.csv'
celestial_dataset = CelestialDataset(read_csv_from_hdfs(csv_path))

#test_csv_path = 'hdfs://okeanos-master:54310/testCUT.csv'
test_csv_path = 'hdfs://okeanos-master:54310/processed_celestial_test.csv'
celestial_test_dataset = CelestialDataset(read_csv_from_hdfs(test_csv_path))



######SIMPLE NN
class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.0.2'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def train_epoch(dataloader, model, criterion, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = criterion(pred, y.view(-1, 1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Loss: ',loss.item())



def test_epoch(dataloader, model, criterion , device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y=y.view(-1, 1)
            pred = model(X)

            test_loss += criterion(pred, y).item() #add the loss value for this batch, will divide later

            pred = torch.round(pred)
            y = y.view(-1)  # Flatten y
            pred = pred.view(-1)
            pred = pred.long()  # Convert to torch.int64

            correct+=torch.sum(pred == y).item()


    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


def train_func(rank, world_size):
    start_time = time.perf_counter()
    epochs = 3
    device="cpu" #we have no gpu in oceanos machines
    setup(rank, world_size)

    train_sampler = DistributedSampler(celestial_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(celestial_dataset, batch_size=128, shuffle=False, sampler=train_sampler)
    load= time.perf_counter()
    print("Load:", load-start_time, "seconds.")
    input_size = len(celestial_dataset.feature_columns)
    model=SimpleClassifier(input_size).to(device)
    model = DistributedDataParallel(model)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(dataloader, model, criterion, optimizer, device)

    #wait for all the processes to reach here
    dist.barrier()
    train=time.perf_counter()
    print("Train:", train-load, "seconds.")
    if rank == 0:
        #only master machine will do the testing
        test_dataloader = DataLoader(celestial_test_dataset, batch_size=128, shuffle=False)
        test_epoch(test_dataloader, model.module, criterion, device)
    print("Done!")
    test=time.perf_counter()
    print("Test:", test-train, "seconds.")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    dist.destroy_process_group()


def main():
    world_size=2
    rank=0
    train_func(rank, world_size)

if __name__ == "__main__":
    main()
    