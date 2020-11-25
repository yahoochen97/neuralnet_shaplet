import torch
import torch.nn as nn

from torch.nn.modules.module import Module
from shapelets_lts.util.utils import get_centroids_of_segments


class SoftMinLayer(nn.Module):
    def __init__(self, shapelets, alpha=-100):
        """
        Args:
            shapelets: (num_shapelet, len_shapelet)
            alpha
        """
        super(SoftMinLayer, self).__init__()
        self.device = shapelets.device
        self.num_shapelet, self.len_shapelet = shapelets.shape # K, L
        self.alpha = alpha
        self.shapelets = nn.Parameter(shapelets) 

    def forward(self, input_seqs):
        n = input_seqs.shape[0]
        min_soft_layers = torch.zeros((n, self.num_shapelet))
        for i in range(n):
            for k in range(self.num_shapelet):
                M = self.dist_soft_min(input_seqs[i].clone(), self.shapelets[k].clone())
                min_soft_layers[i, k] = M
        return min_soft_layers.to(self.device)

    def dist_soft_min(self, input_seq, shapelet):
        Q = input_seq.numel()
        L = self.len_shapelet
        J = Q - L
        M_numerator = 0

        D = torch.zeros(J)
        xi = torch.zeros(J)
        psi = 0
        for j in range(J):
            D[j] = (torch.norm(input_seq[j:j + L].clone() - shapelet,  dim=0))**2/L

        D = D.clone() - torch.min(D.clone())  
        for j in range(J):
            xi[j] = torch.exp(self.alpha * D[j].clone())
            M_numerator += D[j].clone() * xi[j].clone()
            psi += xi[j].clone()
        M = M_numerator / psi
        return M


class Net(nn.Module):
    def __init__(self, data, K, L, device, R=3, alpha=-100):
        super(Net, self).__init__()
        self.device = device
        self.R = R
        print("Initialize shapelets...\n")
        SHAPELETS = []
        for i in range(1, R+1):
            shapelets = get_centroids_of_segments(data, L*i, K)
            shapelets = torch.from_numpy(shapelets).to(device)
            SHAPELETS.append(shapelets)

        # self.shapelets = SHAPELETS
        self.softmins = []

        for i in range(R):
            self.softmins.append(SoftMinLayer(SHAPELETS[i], alpha=alpha))

        self.num_shapelet, self.len_shapelet = K, L
        self.fc1 = nn.Linear(self.num_shapelet*self.R, 24)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(24, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        xs = []
        for i in range(self.R):
            xs.append(self.softmins[i](x))
        #     print(xs[-1].shape)
        # print(x.shape)
        x = torch.cat(xs, dim=1)
        # print(x.shape)
        # x = self.softmin(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x.to(self.device)