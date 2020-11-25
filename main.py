import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from dataloader import UCRDataset
from os.path import expanduser

from yehumodel import Net

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

device = torch.device(dev) 

transform = transforms.Compose([
    transforms.ToTensor()])

transform = None

dataset_name='Gun_Point'

trainset = UCRDataset(dataset_name=dataset_name,
                      dataset_folder='./UCR_TS_Archive_2015/',
                      TYPE="TRAIN",
                      transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                         shuffle=True, num_workers=1)

testset = UCRDataset(dataset_name=dataset_name,
                      dataset_folder='./UCR_TS_Archive_2015/',
                      TYPE="TEST",
                      transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                        shuffle=True, num_workers=1)

torch.autograd.set_detect_anomaly(True)

def main():
    n, m = trainset.data.shape
    K = int(n*0.15)
    L = int(m*0.2)
    R = 3
    alpha = -5
    print(K)
    print(L)

    net = Net(data=trainset.data, K=K, L=L, R=R, device=device, alpha=alpha)

    for p in net.parameters():
                print(p.shape)
                
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=0.1)

    num_epoches = 30
    cost = []

    print("Start training...\n")

    for epoch in range(num_epoches):    
        
        i = 0
        for data in trainloader:

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zeros gradient
            optimizer.zero_grad()
            
            # Forwarding, backpropogation, optimization
            outputs = net(inputs)

            loss = criterion(outputs, labels.long())
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # show cost
            running_loss = loss.item()
            print('[%d, %5d] loss: %.3f' % 
                (epoch + 1, i + 1, running_loss))
            cost.append(running_loss)
            i += 1

    plt.plot(cost)
    plt.ylabel('cost')
    plt.show()

    print("Start testing...\n")

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on test data: %.3f %%' % 
        (100 * correct / total))


if __name__ == '__main__':
    main()

