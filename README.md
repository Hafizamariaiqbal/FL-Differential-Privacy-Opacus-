# FL-Differential-Privacy-Opacus-
This repository contains practical implementation of privacy-preserving Federated Learning using  "Differential Privacy with Opacus"


# Dataset: MNIST
Framework: PyTorch + Opacus + Flower (for FL)

# Install 
pip install torch torchvision opacus flwr

# Code 

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import flwr as fl

# Define model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(5408, 10)
        )

    def forward(self, x):
        return self.net(x)

# Load data
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(train, batch_size=64, shuffle=True), \
           torch.utils.data.DataLoader(test, batch_size=64)

# Client training function with DP
def train(model, train_loader, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
    )

    for epoch in range(1):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# Flower client
class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = CNN().to("cpu")
        self.train_loader, self.test_loader = load_data()

    def get_parameters(self, config): return [val.cpu().numpy() for val in self.model.state_dict().values()]
    def fit(self, parameters, config):
        params_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict(params_dict, strict=True)
        updated_params = train(self.model, self.train_loader, device="cpu")
        return [val.cpu().numpy() for val in updated_params.values()], len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.test_loader.dataset), {}

fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient())




