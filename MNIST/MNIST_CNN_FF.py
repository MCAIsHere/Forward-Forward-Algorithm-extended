import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

def MNIST_loaders(train_batch_size=512, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def overlay_y_on_x(x, y):
    x_ = x.clone()
    mask = torch.zeros_like(x_, device=x_.device)

    lines_coordinates = [6, 10, 14, 18, 22]
    for xi in lines_coordinates:  # rename from x -> xi
        mask[:, xi:xi+2, :] = 1

    if isinstance(y, int):
        y = torch.full((x.size(0),), y, device=x.device)

    # Apply per-example roll
    for i in range(x.size(0)):
        mask[i] = torch.roll(mask[i], shifts=18 * y[i].item(), dims=1)

    x_ = (x_ + mask) / 2
    return x_

class Net(nn.Module):
    def __init__(self, conv_dims, linear_dims):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for c in range(len(conv_dims) - 1):
            self.conv_layers.append(Conv_Layer(conv_dims[c], conv_dims[c + 1]).cuda())
        for l in range(len(linear_dims) - 1):
            self.linear_layers.append(Linear_Layer(linear_dims[l], linear_dims[l + 1]).cuda())

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)

            goodness = []
            for layer in self.conv_layers:
                h = layer(h)
                goodness += [h.pow(2).mean(dim=(1,2,3))]
            h = torch.flatten(h, 1)
            for layer in self.linear_layers:
                h = layer(h)
                goodness += [h.pow(2).mean(dim=1)]

            goodness_per_label += [sum(goodness).unsqueeze(1)]

        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train_net(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.conv_layers):
            print('training conv layer', i, '...')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg)

        h_pos = torch.flatten(h_pos, 1)
        h_neg = torch.flatten(h_neg, 1)

        for i, layer in enumerate(self.linear_layers):
            print('training linear layer', i, '...')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg)


class Linear_Layer(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.relu = nn.ReLU()

        self.threshold = 2.0
        self.num_epochs = 100
        self.opt = Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)
        return self.relu(x_direction @ self.weight.T + self.bias.unsqueeze(0))

    def train_layer(self, x_pos, x_neg):
        for _ in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)

            loss = torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()
            self.opt.zero_grad()

            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

class Conv_Layer(nn.Module):
    def __init__(self, in_features, out_features, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = in_features,
            out_channels = out_features,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(2,2)

        self.threshold = 2.0
        self.num_epochs = 100
        self.opt = Adam(self.parameters(), lr=0.03)

    def forward(self, x):
        norm = x.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-4
        x = x / norm

        x = self.conv(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x

    def train_layer(self, x_pos, x_neg):
        for _ in tqdm(range(self.num_epochs)):
            g_pos = (self.forward(x_pos)).pow(2).mean(dim=(1, 2, 3))
            g_neg = (self.forward(x_neg)).pow(2).mean(dim=(1, 2, 3))

            loss = torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()



if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    net = Net([1, 16, 32],[1568,128,10]).to("cuda")
    x, y = next(iter(train_loader))
    x, y = x.to("cuda"), y.to("cuda")
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0)).to("cuda")
    x_neg = overlay_y_on_x(x, y[rnd])

    net.train_net(x_pos, x_neg)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to("cuda"), y_te.to("cuda")

    test_error = net.predict(x_te).eq(y_te).float().mean().item()
    print('test error:', 1.0 - test_error)
    print(f'Accuracy: {(test_error*100):.2f}%')