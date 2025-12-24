import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

def MNIST_loaders(train_batch_size=500, test_batch_size=1000):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True, transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True, transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(1568,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self, x):
        return self.pipeline(x)

def train():
    nr_epochs = 25

    net.train()
    for epoch in tqdm(range(nr_epochs)):
        epoch_loss = 0.0

        for X, y in train_loader:
            X, y = X.to("cuda"), y.to("cuda")

            predictions = net(X)
            loss = loss_function(predictions, y)

            loss.backward()
            optimizer.step(); optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{nr_epochs}], Loss: {avg_loss:.4f}")

def test():
    net.eval()
    total_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to("cuda"), y.to("cuda")
            predictions = net(X)
            loss = loss_function(predictions, y)
            total_loss += loss.item()

            preds = predictions.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    test_acc = 100 * correct / total
    print(f"Total loss: {total_loss:.4f}, Accuracy: {test_acc:.2f}%")



if __name__ == "__main__":
    torch.manual_seed(1234)

    train_loader, test_loader = MNIST_loaders()
    net = Net().to("cuda")

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.001)

    train()
    test()

    torch.save(net.state_dict(), "CNN_model")