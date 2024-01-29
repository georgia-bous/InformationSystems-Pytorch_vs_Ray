import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim

import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from filelock import FileLock
import pandas as pd
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



def load_data():
    csv_path = r'/home/user/processed_celestial_data.csv'
    test_csv_path = r'/home/user/processed_celestial_test.csv'
    with FileLock("./data.lock"):
        celestial_dataset = CelestialDataset(csv_path)
        celestial_test_dataset = CelestialDataset(test_csv_path)

    return celestial_dataset, celestial_test_dataset
'''

#READ FROM HDFS
class CelestialDataset(Dataset):
    def __init__(self, csv_path, hdfs_host='okeanos-master', hdfs_port=54310):
        hdfs = pa.hdfs.connect(host=hdfs_host, port=hdfs_port)
        with hdfs.open(csv_path, 'rb') as f:
            self.df = pd.read_csv(f)

        self.feature_columns = ['u', 'i', 'psfMag_z','petroRad_g', 'petroRad_r', 'petroRad_i', 'petroRad_z', 'expRad_u', 'expRad_g', 'expRad_r', 'expRad_i', 'expRad_z', 'ra', 'dec', 'l']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = torch.tensor(self.df.loc[idx, self.feature_columns].values, dtype=torch.float32)
        label = torch.tensor(self.df.loc[idx, 'type'], dtype=torch.float32)
        return features, label

@ray.remote
def load_data(hdfs_host='okeanos-master', hdfs_port=54310):
    csv_path = '/processed_celestial_data.csv'
    #csv_path = '/CUT.csv'
    #test_csv_path = '/testCUT.csv'
    test_csv_path = '/processed_celestial_test.csv'
    with FileLock("./data.lock"):
        celestial_dataset = CelestialDataset(csv_path, hdfs_host, hdfs_port)
        celestial_test_dataset = CelestialDataset(test_csv_path, hdfs_host, hdfs_port)

    return celestial_dataset, celestial_test_dataset


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

@ray.remote
def train_epoch(epoch, dataloader, model, criterion, optimizer, device):
    #to synchronize the sampling of data across all processes, if there are more than 1
    if ray.train.get_context().get_world_size() > 1:
        dataloader.sampler.set_epoch(epoch)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = criterion(pred, y.view(-1, 1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Loss: ',loss.item())
    return loss.item()


def test_epoch(dataloader, model, criterion , device):
    model.eval()
    # Divide the dataset size by the world size to get the per-worker dataset size.
    size = len(dataloader.dataset) // ray.train.get_context().get_world_size()
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


def train_func(config:dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size_per_worker = batch_size // ray.train.get_context().get_world_size()
    celestial_dataset, celestial_test_dataset = ray.get(load_data.remote())
    # Create data loaders.
    dataloader = DataLoader(celestial_dataset, batch_size=batch_size_per_worker, shuffle=True)
    test_dataloader = DataLoader(celestial_test_dataset, batch_size=batch_size_per_worker, shuffle=False)
    #prepare the data loader for distributed data sharding
    dataloader= ray.train.torch.prepare_data_loader(dataloader)
    test_dataloader= ray.train.torch.prepare_data_loader(test_dataloader)
    load = time.perf_counter()
    print("Load:", load- start_time, "seconds.")
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    input_size = len(celestial_dataset.feature_columns)
    model=SimpleClassifier(input_size)
    model = ray.train.torch.prepare_model(model)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    '''
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(t, dataloader, model, criterion, optimizer, device)
    '''
    futures = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
          # Call train_epoch.remote and append the future to the list
        future = train_epoch.remote(t, dataloader, model, criterion, optimizer, device)
        futures.append(future)

    # Wait for all tasks to complete
    results = ray.get(futures)
    
    train =time.perf_counter()
    print("Train:", train-load, "seconds.")
    test_epoch(test_dataloader, model, criterion, device)
    test = time.perf_counter()
    print("Test:", test-train, "seconds.")
    print("Done!")

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={"lr": 0.01, "batch_size": 128, "epochs": 3},
    scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
)

start_time = time.perf_counter()
ray.init()
result = trainer.fit()
print(f"Last result: {result.metrics}")
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")