import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from dataloader import UCRDataset
from os.path import expanduser
import sys

from yehumodel import Net, SoftMinLayer

def define_optimizers(net):
    optimizers = [("SGD_Vanilla", optim.SGD(net.parameters(), lr=1)),
                  ("SGD_Momentum", optim.SGD(net.parameters(), lr=1, momentum=0.95)),
                  ("SGD_Nesterov", optim.SGD(net.parameters(), lr=1, momentum=0.95, nesterov=True)),
                  ("Adam", optim.Adam(net.parameters(), lr=0.1)),
                  ("LBFGS", optim.LBFGS(net.parameters(), lr=1))
                 ]

    return optimizers


def main(trainset, testset, dataset_name):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)

    n, m = trainset.data.shape

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=n,
                                            shuffle=True, num_workers=1)
                                            
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=True, num_workers=1)

    
    K = int(n * 0.15)
    L = int(m * 0.2)
    R = 3
    lambda_reg = 1e-4
    alpha = -10

    net = Net(data=trainset.data, K=K, L=L, R=R, device=device, alpha=alpha)

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    num_epoches = 50
    
    # optimizer = optim.Adam(net.parameters(), lr=0.1)
    optimizers = define_optimizers(net)
    
    train_loss = np.zeros((len(optimizers), num_epoches))

    print("Start training...\n")

    for i, optimizer in enumerate(optimizers):

        for epoch in range(num_epoches):

            for data in trainloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zeros gradient
                optimizer.zero_grad()

                # Forwarding, backpropogation, optimization
                outputs = net(inputs)

                loss = criterion(outputs, labels.long())
                all_linear1_params = torch.cat([x.view(-1) for x in net.fc1.parameters()])
                all_linear2_params = torch.cat([x.view(-1) for x in net.fc2.parameters()])

                loss += lambda_reg * torch.norm(all_linear1_params, 2) + lambda_reg * torch.norm(all_linear2_params, 2)

                loss.backward()

                # Update parameters
                optimizer.step()

                # show cost
                running_loss = loss.item()
                print('[%d] loss: %.3f' %
                    (epoch + 1, running_loss))
            
            train_loss[i, epoch] = running_loss

        plt.plot(train_loss[i])
        plt.ylabel('training loss')
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

    if len(sys.argv)==1:
        dataset_name = 'Gun_Point'
    else:
        dataset_name = sys.argv[1]

    print(dataset_name)

    trainset = UCRDataset(dataset_name=dataset_name,
                        dataset_folder='./UCR_TS_Archive_2015/',
                        TYPE="TRAIN")

    testset = UCRDataset(dataset_name=dataset_name,
                        dataset_folder='./UCR_TS_Archive_2015/',
                        TYPE="TEST")

    torch.autograd.set_detect_anomaly(True)

    main(trainset, testset, dataset_name)