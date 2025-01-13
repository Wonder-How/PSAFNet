import numpy as np
import torch
from scipy.io import loadmat
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset, random_split

def extract_segments_eeg(eeg_data, window_length=1, fs=200):
    no_target_start = int(1 * fs)
    no_target_end = int((1 + window_length) * fs)
    target_start = int(5 * fs)
    target_end = int((5 + window_length) * fs)

    no_target_data = eeg_data[:, no_target_start:no_target_end, :]
    target_data = eeg_data[:, target_start:target_end, :]

    no_target_data_reshaped = no_target_data.reshape(no_target_data.shape[2], -1)
    target_data_reshaped = target_data.reshape(target_data.shape[2], -1)

    no_target_labels = np.zeros(no_target_data_reshaped.shape[0])
    target_labels = np.ones(target_data_reshaped.shape[0])

    X = np.concatenate([no_target_data_reshaped, target_data_reshaped], axis=0)
    y = np.concatenate([no_target_labels, target_labels], axis=0)

    X = X.reshape(X.shape[0], eeg_data.shape[0], no_target_data.shape[1])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X, y = shuffle(X, y, random_state=42)
    return X, y

def extract_segments_path(file_path, window_length=1, fs=200, init_time=1.0, target_time=5.0):
    data = loadmat(file_path)
    eeg_data = data['EEG']['data'][0][0]

    no_target_start = int(init_time * fs)
    no_target_end = int((init_time + window_length) * fs)
    target_start = int(target_time * fs)
    target_end = int((target_time + window_length) * fs)

    no_target_data = eeg_data[:, no_target_start:no_target_end, :]
    target_data = eeg_data[:, target_start:target_end, :]

    no_target_data = np.transpose(no_target_data, (2, 0, 1))
    target_data = np.transpose(target_data, (2, 0, 1))

    no_target_data_reshaped = no_target_data.reshape(no_target_data.shape[0], -1)
    target_data_reshaped = target_data.reshape(target_data.shape[0], -1)

    no_target_labels = np.zeros(no_target_data_reshaped.shape[0])
    target_labels = np.ones(target_data_reshaped.shape[0])

    X = np.concatenate([no_target_data_reshaped, target_data_reshaped], axis=0)
    y = np.concatenate([no_target_labels, target_labels], axis=0)

    X = X.reshape(X.shape[0], eeg_data.shape[0], no_target_data.shape[2])
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    return X, y

def create_dataloaders(X, y, batch_size=16, train_ratio=0.8, val_ratio=0.1):
    dataset = TensorDataset(X, y)
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def create_dataloaders_folds(X_train, y_train, X_val, y_val, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def create_cross_subject_dataloaders(file_paths, window_length=1, batch_size=16, test_index=7):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    val_index = len(file_paths) - 1 if test_index != len(file_paths) - 1 else len(file_paths) - 2

    for i, file_path in enumerate(file_paths):
        X, y = extract_segments_path(file_path, window_length=window_length, init_time=1, target_time=5.0)

        if i == test_index:
            X_test, y_test = X, y
        elif i == val_index:
            X_val, y_val = X, y
        else:
            X_train.append(X)
            y_train.append(y)

    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
