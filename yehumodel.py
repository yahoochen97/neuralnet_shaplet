import torch
import torch.nn as nn

from utils import get_centroids_of_segments

class SoftMinLayer(nn.Module):
    def __init__(self, shapelets, alpha=-100):
        """
        Args:
            shapelets: (num_shapelet, len_shapelet)
            alpha
        """
        super(SoftMinLayer, self).__init__()
        self.device = shapelets.device
        self.num_shapelet, self.len_shapelet = shapelets.shape  # K, L
        self.alpha = alpha
        self.shapelets = nn.Parameter(shapelets)

    def forward(self, input_seqs):
        n = input_seqs.shape[0]
        min_soft_layers = torch.zeros((n, self.num_shapelet))

        for k in range(self.num_shapelet):
            M = self.dist_soft_min(input_seqs, self.shapelets[k])
            min_soft_layers[:, k] = M
        return min_soft_layers.to(self.device)

    def dist_soft_min(self, input_seq, shapelet):
        n=input_seq.shape[0]
        Q = input_seq.shape[1]
        L = self.len_shapelet
        J = Q - L+1
        # print("Q"+str(Q)+"\n"+"L"+str(L))
        M_numerator = torch.zeros(n)

        D = torch.zeros((n,J))
        xi = torch.zeros((n,J))
        psi = torch.zeros(n)
        for j in range(J):
            D[:,j] = (torch.norm(input_seq[:,j:j + L] - shapelet, dim=1)) ** 2 / L

        D = D - torch.min(D)
        for j in range(J):
            xi[:,j] = torch.exp(self.alpha * D[:,j])
            M_numerator = M_numerator.clone()+ D[:,j].clone() * xi[:,j].clone()
            psi =psi.clone()+ xi[:,j].clone()
        M = M_numerator / psi
        return M



class Net(nn.Module):
    def __init__(self, data, C, K, L, device, R=3, alpha=-100):
        super(Net, self).__init__()
        self.device = device
        self.R = R
        self.alpha=alpha
        self.C = C
        print("Initialize shapelets...\n")
        SHAPELETS = []
        for i in range(1, R + 1):
            shapelets = get_centroids_of_segments(data, L * i, K)
            shapelets = torch.from_numpy(shapelets).to(device)
            SHAPELETS.append(shapelets)

        # self.shapelets = SHAPELETS
        # self.softmins = []
        #
        # for i in range(R):
        #     self.softmins.append(SoftMinLayer(SHAPELETS[i], alpha=alpha))
        #self.softminlayer=Softmini(SHAPELETS,self.alpha,self.R)
        self.smls = nn.ModuleList([SoftMinLayer(SHAPELETS[i], alpha=alpha) for i in range(R)])
        self.num_shapelet, self.len_shapelet = K, L
        self.fc1 = nn.Linear(self.num_shapelet * self.R, 24)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(24, self.C)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # for i in range(self.R):
        #     xs.append(self.softmins[i](x))
        #     print(xs[-1].shape)
        # print(x.shape)
        # x = self.softmin(x)
        # x=self.softminlayer(x)
        for i,m in enumerate(self.smls):
            if i>=1:
                # print(x.shape)
                x1 = torch.cat([x1.double(),m(x).double()],dim=1)
            else:
                x1=m(x)

        x1 = self.fc1(x1.float())
        x1 = self.activation(x1)
        x1 = self.fc2(x1)
        x1 = self.softmax(x1)

        return x1.to(self.device)

