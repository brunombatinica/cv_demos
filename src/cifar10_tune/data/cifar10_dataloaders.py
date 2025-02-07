import torch
import pickle
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


# reading in data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

## define datasets 
class ImageDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data).reshape(-1,3,32,32) / 255 # will make it float32
        self.y_data = torch.tensor(y_data).to(torch.long) # will make it int64
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        x = self.x_data[idx] #dont want to squeeze here as we want to keep the [CxHxW] even if one 1 channel
        y = self.y_data[idx]
        return x, y


class FlatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data) / 255 # will make it float32
        self.y_data = torch.tensor(y_data).to(torch.long) # will make it int64
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx].view(1,-1).squeeze() 
        y = self.y_data[idx]
        return x, y
    
def get_cifar10_dataloaders(batch_size=32):
    '''
    function to load cifar10 data and return train and test dataloaders
    args:
        batch_size: int, default=32
    returns:
        train_dataloader: DataLoader, train dataloader
        test_dataloader: DataLoader, test dataloader
    '''
    # load data
    x_train = []
    y_train = []
    for i in range(1,6):
        path = "C:/Users/bruno/OneDrive/Documents/Code/projects/cv_demos/data"
        batch = unpickle(os.path.join(path,"cifar-10-batches-py",f"data_batch_{i}"))
        x_train.append(batch["data"])
        y_train.append(batch["labels"])


    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    test = unpickle(os.path.join(path,"cifar-10-batches-py","test_batch"))
    x_test = test["data"]
    y_test = np.array(test["labels"])

    # Create datset and dataloader
    train_dataset = ImageDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # validation set - FIX THIS TO USE SUBSET OF TRAIN SET
    val_dataset = ImageDataset(x_test, y_test)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # # test set
    # test_dataset = ImageDataset(x_test, y_test)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader


