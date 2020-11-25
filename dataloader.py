from os import path
from os import listdir
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch


class UCRDataset(Dataset):
    """UCR dataset."""

    def __init__(self, dataset_name, dataset_folder, TYPE, transform=None):
        """
        Args:
            dataset_name
            dataset_folder
            TYPE: TRAIN or TEST
            transform
        """
        dataset_path = path.join(dataset_folder, dataset_name)
        self.dataset_path = dataset_path
        if TYPE=="TRAIN":
            self.file_path = path.join(dataset_path, '{}_TRAIN'.format(dataset_name))
        else:
            self.file_path = path.join(dataset_path, '{}_TEST'.format(dataset_name))

        self.raw_arr = genfromtxt(self.file_path, delimiter=',')
        self.data = self.raw_arr[:, 1:]
        self.labels = self.raw_arr[:, 0]
        if np.min(self.labels)==1:
            self.labels -= 1
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.transform:
            samples = (self.transform(self.data[idx]), self.transform(self.labels[idx]))
        else:
            samples = (self.data[idx], self.labels[idx])

        return samples

    def getitem(self, idx):
        return self.__getitem__(idx)