import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

def MNIST_loaders(train_batch_size=50, test_batch_size=1000):
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
    batch_size = x.size(0)
    label_channel = torch.zeros_like(x)
    for i in range(batch_size):
        label_channel[i] = y[i].float() / 9.0 
    return torch.cat([x, label_channel], dim=1)

class Net(nn.Module):
    def __init__(self, conv_dims):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for c in range(len(conv_dims) - 1):
            self.conv_layers.append(Conv_Layer(conv_dims[c], conv_dims[c + 1]).cuda())

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)

            goodness = []
            for layer in self.conv_layers:
                h = layer(h)
                goodness += [h.pow(2).mean(dim=(1,2,3))]

            goodness_per_label += [sum(goodness).unsqueeze(1)]

        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train_net(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.conv_layers):
            print('training conv layer', i, '...')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg)

class Conv_Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_features, out_channels = out_features, 
                              kernel_size = 7,stride = 1,padding = 3)
        self.norm = nn.LayerNorm([out_features, 28, 28])
        self.relu = nn.ReLU()
        self.threshold = 28*28*out_features
        self.num_epochs = 200
        self.opt = Adam(self.parameters(), lr=0.0005)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

    def train_layer(self, x_pos, x_neg):
        for _ in tqdm(range(self.num_epochs)):
            g_pos = (self.forward(x_pos)).pow(2).sum(dim=(1,2,3)) - self.threshold
            g_neg = -(self.forward(x_neg)).pow(2).sum(dim=(1,2,3)) + self.threshold

            # Binary Cross Entropy loss
            loss = torch.log(1 + torch.exp(-g_pos)).mean() + torch.log(1 + torch.exp(-g_neg)).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()



if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    net = Net([2, 128, 128, 128]).to("cuda")
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

    torch.save(net.state_dict(), "CNN_FF_model")