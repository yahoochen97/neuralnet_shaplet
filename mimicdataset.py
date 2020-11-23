# Customized Mimic dataset

from torch.utils.data import Dataset
import torch
import urllib.request
import zipfile
import pandas as pd
import numpy as np


class MimicDataset(Dataset):
    """Mimic dataset."""

    def __init__(self, zip_path, feature_name, max_input_length, patientfile_name="PATIENTS.csv"):
        """
        Args:
            zip_path (string): path of zipped dataset.
            feature_name (string): name of feature to be extracted.
            max_input_length (string): trimmed length of input sequence
            patientfile_name (string): file name of the patients.csv file
        """
        self.zip_path = zip_path
        self.__update_feature__(feature_name)
        self.max_input_length = max_input_length
        self.patientfile_name = patientfile_name

    def update_feature(self, feature_name):
        """
        Args:
            feature_name (string): name of feature to be extracted.
        """
        self.__update_feature__(feature_name)

    def __update_feature__(self, feature_name):
        """
        Args:
            feature_name (string): name of feature to be extracted.
        """
        self.feature_name = feature_name

        # read csv file names in zipped file
        archive = zipfile.ZipFile(self.zip_path, 'r')
        fileList = archive.namelist()
        csvfile_names = [filename for filename in fileList  if filename.endswith(".csv") and not filename.startswith("P") ]
        self.csvfile_names = []

        # iterate csv files and select those containing the feature
        for csvfile_name in csvfile_names:
            df = pd.read_csv(archive.open(csvfile_name)) 
            if self.feature_name in df:
                self.csvfile_names.append(csvfile_name)

    def __len__(self):
        return len(self.csvfile_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        archive = zipfile.ZipFile(self.zip_path, 'r')

        samples = np.zeros((self.__len__(), self.max_input_length))
        for i in idx:
            df = pd.read_csv(archive.open(self.csvfile_names[i])) 
            sample = self.__preprocess__(df)
            samples[i] = sample.T

        return samples

    def getitem(self, idx):
        return self.__getitem__(idx)

    def __preprocess__(self, df):
        """
        Args:
            df: raw dataframe to be processed 
        """
        sample = df[self.feature_name]
        data = self.__remove_outlier__(sample)

        sample = np.zeros((self.max_input_length,))
        if data.shape[0]<self.max_input_length:
            # append with zeros
            sample[-data.shape[0]:] = data
        else:
            # trim data to input length
            sample = data[data.shape[0]-self.max_input_length:]

        return sample

    def __remove_outlier__(self, data):
        """
        Args:
            data (numpy array): raw array before removing outliers
        """
        tmp = data
        m = np.mean(tmp)
        sigma = np.std(tmp)
        data = data[np.abs((tmp-m)/sigma)<3]

        begin_value = data[0]
        n = data.shape[0]
        end_value = data[-1]
        i = 1
        while i < n:
            if data[i] == begin_value:
                i = i + 1
            else:
                break
        
        j = n - 2
        while j > 0:
            if data[j] == end_value:
                j = j - 1
            else:
                break

        data = data[i:j]

        return data

