import ray
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from torchvision.datasets.mnist import read_image_file, read_label_file
import time
import os
from torchvision.transforms import ToPILImage
import pyarrow as pa
import numpy as np

'''
#READ FROM LOCAL
class CustomEMNIST(Dataset):
    def __init__(self, train, transform):
        """
        train: If True, load the training data, else load the test data.
        transform: Optional transform to be applied on a sample.
        """
        self.transform = transform
        root = os.path.expanduser('~/EMNIST/raw')
        if train:
            self.images = read_image_file(os.path.join(root, 'emnist-digits-train-images-idx3-ubyte'))
            self.labels = read_label_file(os.path.join(root, 'emnist-digits-train-labels-idx1-ubyte'))
        else:
            self.images = read_image_file(os.path.join(root, 'emnist-digits-test-images-idx3-ubyte'))
            self.labels = read_label_file(os.path.join(root, 'emnist-digits-test-labels-idx1-ubyte'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = ToPILImage()(image)  # Convert tensor to PIL Image
        if self.transform:
            image = self.transform(image)
        return image, label
'''

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

@ray.remote
def train(model, device, train_loader, optimizer, epoch, criterion):
    if ray.train.get_context().get_world_size() > 1:
        train_loader.sampler.set_epoch(epoch)
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

@ray.remote
def test(model, device, test_loader):
    model.eval()
    size = len(test_loader.dataset)// ray.train.get_context().get_world_size()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= size
    accuracy = 100. * correct / size
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{size} ({accuracy:.0f}%)')


def train_func(config:dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    device="cpu"
    batch_size_per_worker = batch_size // ray.train.get_context().get_world_size()
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))  # Normalize the tensors
    ])

    train_data = CustomEMNIST(train=True, transform=transform)
    test_data = CustomEMNIST(train=False, transform=transform)
    '''
    train_data = datasets.EMNIST(root='./', split='digits', train=True, download=True, transform=transform)
    test_data = datasets.EMNIST(root='./', split='digits', train=False, download=True, transform=transform)
    '''
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
    train_loader= ray.train.torch.prepare_data_loader(train_loader)

    model = Net()
    model = ray.train.torch.prepare_model(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.to(device)

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch, criterion)

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    test_loader= ray.train.torch.prepare_data_loader(test_loader)
    test(model, device, test_loader)


if __name__=="__main__":
    trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    train_loop_config={"lr": 0.01, "batch_size": 64, "epochs": 3},
    scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
    )

    start_time = time.perf_counter()
    ray.init()
    result = trainer.fit()
    print(f"Last result: {result.metrics}")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")