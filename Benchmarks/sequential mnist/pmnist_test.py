import torch as t
import torch.nn.functional as F
import sys
sys.path.append("../")
from utils import data_generator
from model import CL_RNN
import os

if t.cuda.is_available():
    device = t.device('cuda')
else:
    device = t.device('cpu')

root = 'path to data'
t.cuda.manual_seed(1111)
batch_size = 64
n_classes = 10
input_channels = 1
seq_length = int(784 / input_channels)
epochs = 12
steps = 0

train_loader, test_loader = data_generator(root, batch_size)
channel_sizes = [1] * 8
kernel_size = 7
model = CL_RNN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=0.05).to(device=device)
load = 0
if os.path.exists('checkpoint.pt'):
    model.load_state_dict(t.load('checkpoint.pt'))
    load = 1

lr = 2e-3
optimizer = t.optim.Adam(model.parameters(), lr=lr)


def train(ep):
    global steps
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device=device), target.to(device=device)
        data = data.view(-1, input_channels, seq_length)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss.item() / 100, steps))
            train_loss = 0


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with t.no_grad():
        for data, target in test_loader:
            data, target = data.to(device=device), target.to(device=device)
            data = data.view(-1, input_channels, seq_length)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return test_loss


if load == 0:
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()
        t.save(model.state_dict(), 'checkpoint.pt')
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
else:
    test()
