from typing import ForwardRef
import torch
from torch._C import device
import torch.nn as nn ## neural networks 
import torch.nn.functional as F
from torch.nn.modules.loss import TripletMarginWithDistanceLoss ## same as nn but with functional way
import torch.optim as optim ## contains optimizers
from torchvision import datasets, transforms


### Define the model 
#three layers (500, 1000, 10)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500) # since pictures are 28X28 pixels input and output neurons
        self.fc2 = nn.Linear(500, 1000) # next layer gets inout from previous output
        self.fc3 = nn.Linear(1000, 10) # 10 cause we have 10 digits

    def forward(self, x):
        x = x.view(-1, 784) # reshape it, to fit view function
        x = F.relu(self.fc1(x)) #applyin activation function for non linearity
        x = F.relu(self.fc2(x))
        x = self.fc3(x) #linear function
        return F.log_softmax(x, dim=1) # output will be in between 0:1


### Load the Data
# load training data normalize pixel data, 
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data',train=True, download=True,
                        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081))])), batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data',train=False,
                        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081))])), batch_size=1000, shuffle=True)



### Training and testin Loops


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target =data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx%100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset),
            100. *batch_idx/len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad(): ## don't remember gradients
        for data, target in test_loader:
            data, target =data.to(device), target.to(device)
            output=model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # count batch loss
            pred =output.argmax(dim=1, keepdim=True) # get the index of  max of log probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))



### Testing
use_cuda = torch.cuda.is_available()
torch.manual_seed(42) # random weights initializartion
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device) # initialaze device
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5) # what it;s optimizing all the parameteres based on forward function

test(model, device, test_loader)
for epoch in range (1, 3+1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
torch.save(model.state_dict(), "mnist.pt")

