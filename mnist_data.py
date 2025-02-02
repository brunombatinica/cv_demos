import os
import struct
import numpy as np
import torch
from torchvision import datasets

# Define data path
DATA_DIR = os.getcwd() + "/data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load MNIST from TorchVision
train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True)
test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True)

# Function to save as IDX format
def save_idx_file(filename, data, is_image=True):
    with open(filename, "wb") as f:
        if is_image:
            magic_number = 2051  # Magic number for images
            num_items, rows, cols = data.shape
            header = struct.pack(">IIII", magic_number, num_items, rows, cols)
        else:
            magic_number = 2049  # Magic number for labels
            num_items = data.shape[0]
            header = struct.pack(">II", magic_number, num_items)

        f.write(header)
        f.write(data.tobytes())

# Convert images & labels to NumPy arrays
train_images = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()
test_images = test_dataset.data.numpy()
test_labels = test_dataset.targets.numpy()

# Save images and labels in IDX format
save_idx_file(os.path.join(DATA_DIR, "train-images-idx3-ubyte"), train_images)
save_idx_file(os.path.join(DATA_DIR, "train-labels-idx1-ubyte"), train_labels, is_image=False)
save_idx_file(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte"), test_images)
save_idx_file(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte"), test_labels, is_image=False)

print("MNIST dataset saved in ~/data in original format!")