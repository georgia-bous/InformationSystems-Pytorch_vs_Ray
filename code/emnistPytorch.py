import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import os
import torch.distributed as dist
import time
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
import pyarrow as pa
import numpy as np


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '192.168.0.2'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


#READ FROM HDFS
class CustomEMNIST(Dataset):
    def __init__(self, train, transform, hdfs_host='okeanos-master', hdfs_port=54310):
        self.transform = transform
        hdfs_root = 'hdfs:///'

        hdfs = pa.hdfs.connect(host=hdfs_host, port=hdfs_port)

        if train:
            image_path = os.path.join(hdfs_root, 'emnist-digits-train-images-idx3-ubyte')
            label_path = os.path.join(hdfs_root, 'emnist-digits-train-labels-idx1-ubyte')
        else:
            image_path = os.path.join(hdfs_root, 'emnist-digits-test-images-idx3-ubyte')
            label_path = os.path.join(hdfs_root, 'emnist-digits-test-labels-idx1-ubyte')

        with hdfs.open(image_path, 'rb') as f:
            self.images = CustomEMNIST.read_idx3_ubyte(f)
        with hdfs.open(label_path, 'rb') as f:
            self.labels = CustomEMNIST.read_idx1_ubyte(f)

    def read_idx3_ubyte(file):
        file.read(4)  # Skip magic number
        num_images = int.from_bytes(file.read(4), byteorder='big')
        num_rows = int.from_bytes(file.read(4), byteorder='big')
        num_cols = int.from_bytes(file.read(4), byteorder='big')
        images = np.frombuffer(file.read(), dtype=np.uint8)
        images = images.reshape(num_images, 1, num_rows, num_cols)
        return torch.tensor(images, dtype=torch.float)


    def read_idx1_ubyte(file):
        file.read(4)  # Skip magic number
        num_labels = int.from_bytes(file.read(4), byteorder='big')
        labels = np.frombuffer(file.read(), dtype=np.uint8)
        return torch.tensor(labels, dtype=torch.long)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].float() /  255.0  # Scale images to [0, 1]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # EMNIST images are 28x28
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # Output layer for 10 digits

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the images
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch, criterion):
    size = len(train_loader.dataset)
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        pred = model(data)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{size} ({100. * batch_idx * len(data) / size:.0f}%)]\tLoss: {loss.item():.6f}")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')


def main():
    start_time=time.perf_counter()
    device="cpu"
    epochs=3
    rank=0
    world_size=2

    setup(rank, world_size)
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))  # Normalize the tensors
    ])
    '''
    #LOAD FROM LOCAL
    train_data = datasets.EMNIST(root='./', split='digits', train=True, download=True, transform=transform)
    test_data = datasets.EMNIST(root='./', split='digits', train=False, download=True, transform=transform)
    '''
    train_data = CustomEMNIST(train=True, transform=transform)
    test_data = CustomEMNIST(train=False, transform=transform)

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, sampler=train_sampler)

    model = Net()
    model = DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.to(device)

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch, criterion)

    dist.barrier()
    if rank == 0:
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
        test(model, device, test_loader)

    end_time=time.perf_counter()
    print("Elapsed:", end_time-start_time, "seconds.")
    dist.destroy_process_group()

if __name__=="__main__":
    main()