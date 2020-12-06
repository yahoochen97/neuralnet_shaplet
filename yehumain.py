import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import UCRDataset
import os
from os.path import expanduser
import sys

from yehumodel import Net, SoftMinLayer

def define_optimizers():
    opt_names = ["SGD_Vanilla",
                "SGD_Momentum",
                "SGD_Nesterov",
                "Adam"]

    return opt_names


def plot_train_loss():
    result_path = "./results/"
    suffix = "_train_loss.csv"
    opt_names = define_optimizers()
    filenames = os.listdir(result_path )
    filenames = [ filename for filename in filenames if filename.endswith(suffix) ]
    dataset_names = np.array([ filename[:-15] for filename in filenames]).reshape((4,-1))
    p,q = dataset_names.shape

    fig, axs = plt.subplots(p, q)

    for i in range(p):
        for j in range(q):
            dataset_name = dataset_names[i, j]
            losses = pd.read_csv(result_path+dataset_name+suffix, index_col=False)
            for opt_name in opt_names:
                loss = losses[opt_name]
                axs[i, j].plot(range(len(loss)), loss, label=opt_name)
            axs[i, j].set_title(dataset_name)
            axs[i, j].legend(loc=0, prop={'size': 6})
    fig.tight_layout()
    plt.savefig("train_losses.png", dpi=400)


def plot_test_acc():
    result_path = "./results/"
    suffix = "_test_acc.csv"
    opt_names = define_optimizers()
    filenames = os.listdir(result_path )
    filenames = [ filename for filename in filenames if filename.endswith(suffix) ]
    dataset_names = np.array([ filename[:-13] for filename in filenames]).reshape((4,-1))
    p,q = dataset_names.shape

    fig, axs = plt.subplots(p, q)

    for i in range(p):
        for j in range(q):
            dataset_name = dataset_names[i, j]
            accs = pd.read_csv(result_path+dataset_name+suffix, index_col=False)
            axs[i, j].barh(opt_names, accs.loc[0])
            for k, v in enumerate(accs.loc[0]):
                axs[i, j].text(x=v+0.01, y=k-0.25, s=str(round(v,3)))
            axs[i, j].set_xlim(0, 1.3)
            axs[i, j].set_title(dataset_name)

    fig.tight_layout()
    plt.savefig("test_accs.png", dpi=400)


def main(trainset, testset, dataset_name):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    device = torch.device(dev)

    n, m = trainset.data.shape
    C = np.unique(trainset.labels).shape[0]
    print(C)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=1)
                                            
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=True, num_workers=1)

    
    K = int(n * 0.15)
    L = int(m * 0.2)
    R = 3
    lambda_reg = 1e-4
    alpha = -5

    criterion = nn.CrossEntropyLoss()
    num_epoches = 200
    
    # optimizer = optim.Adam(net.parameters(), lr=0.1)
    opt_names = define_optimizers()
    
    train_loss = np.zeros((num_epoches, len(opt_names)))
    test_acc = np.zeros((1, len(opt_names)))

    for i, opt_name in enumerate(opt_names):

        net = Net(data=trainset.data, C=C, K=K, L=L, R=R, device=device, alpha=alpha)
        net.to(device)

        if opt_name=="SGD_Vanilla":
            optimizer = optim.SGD(net.parameters(), lr=1)
        elif opt_name=="SGD_Momentum":
            optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.98)
        elif opt_name=="SGD_Nesterov":
            optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.98, nesterov=True)
        else:
            optimizer = optim.Adam(net.parameters(), lr=0.01)

        print(opt_names[i])

        print("Start training...\n")

        for epoch in range(num_epoches):

            for data in trainloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)

                loss = criterion(outputs, labels.long())
                all_linear1_params = torch.cat([x.view(-1) for x in net.fc1.parameters()])
                all_linear2_params = torch.cat([x.view(-1) for x in net.fc2.parameters()])

                loss += lambda_reg * torch.norm(all_linear1_params, 2) + lambda_reg * torch.norm(all_linear2_params, 2)

                loss.backward()
                optimizer.step()

            inputs, labels = torch.from_numpy(trainset.data), torch.from_numpy(trainset.labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)

            running_loss = criterion(outputs, labels.long())
            all_linear1_params = torch.cat([x.view(-1) for x in net.fc1.parameters()])
            all_linear2_params = torch.cat([x.view(-1) for x in net.fc2.parameters()])

            running_loss += lambda_reg * torch.norm(all_linear1_params, 2) + lambda_reg * torch.norm(all_linear2_params, 2)

            print('[%d] loss: %.3f' %
                (epoch + 1, running_loss))
            
            train_loss[epoch, i] = running_loss

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

        acc = (100 * correct / total)
        print('Accuracy of the network on test data: %.3f %%' %
          acc)

        test_acc[0, i] = acc/100

    np.savetxt("results/"+dataset_name+"_train_loss.csv", train_loss, delimiter=',', header=",".join(opt_names), comments="", fmt='%1.3f')
    # df = pd.DataFrame(data=train_loss, columns=opt_names)
    # df.to_csv(, index=False)

    np.savetxt("results/"+dataset_name+"_test_acc.csv", test_acc, delimiter=',', header=",".join(opt_names), comments="", fmt='%1.3f')

    # df = pd.DataFrame(data=test_acc, columns=opt_names)
    # df.to_csv("results/"+dataset_name+"_test_acc.csv", index=False)

    # plt.plot(train_loss[i])
    # plt.ylabel('training loss')
    # plt.show()


if __name__ == '__main__':

    dataset_folder='./UCR_TS_Archive_2015/'

    if len(sys.argv)==1:
        # for dataset_name in os.listdir(dataset_folder):
        #     trainset = UCRDataset(dataset_name=dataset_name,
        #                 dataset_folder=dataset_folder,
        #                 TYPE="TRAIN")
        #     C = np.unique(trainset.labels).shape[0]
        #     print(dataset_name + " " + str(C))
        plot_train_loss()
        plot_test_acc()
        exit()
    else:
        dataset_name = sys.argv[1]

    print(dataset_name)

    trainset = UCRDataset(dataset_name=dataset_name,
                        dataset_folder=dataset_folder,
                        TYPE="TRAIN")

    testset = UCRDataset(dataset_name=dataset_name,
                        dataset_folder=dataset_folder,
                        TYPE="TEST")

    torch.autograd.set_detect_anomaly(True)

    main(trainset, testset, dataset_name)