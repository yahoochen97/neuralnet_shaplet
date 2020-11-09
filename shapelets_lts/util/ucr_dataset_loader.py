from os import path
from os import listdir
import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def remove_outlier(data):
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


def load_mimic(dataset_name, dataset_folder, feature_names, MAX_LENGTH):
    dataset_path = path.join(dataset_folder, dataset_name)
    filenames = listdir(dataset_path)
    csvfilenames = [ filename for filename in filenames if filename.endswith(".csv") and not filename.startswith("P") ]
    valid_csvfiles = []
    for csvfilename in csvfilenames:
        onedatapath = path.join(dataset_path, csvfilename)
        onedata = pd.read_csv(onedatapath)
        have_all_feature = True
        for feature_name in feature_names:
            if feature_name not in onedata.columns or np.sum(onedata.loc[:, feature_name])==0:
                have_all_feature = False
        if have_all_feature:
            valid_csvfiles.append(csvfilename)
    
    for feature_name in feature_names:
        data = np.zeros((len(valid_csvfiles), MAX_LENGTH))
        for i, valid_csvfile in enumerate(valid_csvfiles):
            onedatapath = path.join(dataset_path, valid_csvfile)
            onedata = pd.read_csv(onedatapath)
            # print(onedata.shape)
            onedata = onedata[[feature_name]].values
            onedata = remove_outlier(onedata)
            # print(onedatapath)
            if onedata.shape[0]<MAX_LENGTH:
                tmp = onedata
                tmp = (tmp - np.mean(tmp))/np.std(tmp)
                data[i, -onedata.shape[0]:] = tmp 
            else:
                tmp = onedata[onedata.shape[0]-MAX_LENGTH:]
                tmp = (tmp - np.mean(tmp))/np.std(tmp)
                data[i] = tmp
            # print(data[i])
        
    patientdata =  pd.read_csv(path.join(dataset_path, 'PATIENTS.csv'))

    labels = np.zeros((len(valid_csvfiles),))

    for i, valid_csvfile in enumerate(valid_csvfiles):
        patientid = int(valid_csvfile[0:-4])
        labels[i] = patientdata.loc[patientdata.SUBJECT_ID==patientid, "EXPIRE_FLAG"]

    # split training data and test_data
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, stratify=labels, test_size=0.2)
    
    return train_data, train_labels, test_data, test_labels


def load_dataset(dataset_name, dataset_folder):
    dataset_path = path.join(dataset_folder, dataset_name)
    train_file_path = path.join(dataset_path, '{}_TRAIN'.format(dataset_name))
    test_file_path = path.join(dataset_path, '{}_TEST'.format(dataset_name))

    # training data
    train_raw_arr = genfromtxt(train_file_path, delimiter=',')
    train_data = train_raw_arr[:, 1:]
    train_labels = train_raw_arr[:, 0] - 1
    # one was subtracted to change the labels to 0 and 1 instead of 1 and 2

    # test_data
    test_raw_arr = genfromtxt(test_file_path, delimiter=',')
    test_data = test_raw_arr[:, 1:]
    test_labels = test_raw_arr[:, 0] - 1

    return train_data, train_labels, test_data, test_labels
