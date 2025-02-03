import os
import shutil
from torchvision import datasets

# import struct
# import numpy as np
# import torch

# Define data path
DATA_DIR = os.getcwd() + "/data"
MNIST_RAW_DIR = os.path.join(DATA_DIR, "MNIST", "raw")

os.makedirs(DATA_DIR, exist_ok=True)

# Load MNIST from TorchVision
train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True) 
test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True)

# List of files to move
files_to_move = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte"
]

# Move each unzipped file
for filename in files_to_move:
    src = os.path.join(MNIST_RAW_DIR, filename)
    dst = os.path.join(DATA_DIR, filename)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved {filename} to {DATA_DIR}")






# def save_idx_images(filename, images):
#     """
#     Save images in IDX format (as original MNIST .idx3-ubyte).
#     """
#     num_samples, height, width = images.shape

#     # Open file in binary mode
#     with open(filename, "wb") as f:
#         # Write magic number (2051 for images)
#         f.write(struct.pack(">I", 2051))
#         # Write number of images
#         f.write(struct.pack(">I", num_samples))
#         # Write image dimensions (28x28)
#         f.write(struct.pack(">I", height))
#         f.write(struct.pack(">I", width))
#         # Write pixel data (flatten images)
#         f.write(images.astype(np.uint8).tobytes())

# def save_idx_labels(filename, labels):
#     """
#     Save labels in IDX format (as original MNIST .idx1-ubyte).
#     """
#     num_samples = labels.shape[0]

#     with open(filename, "wb") as f:
#         # Write magic number (2049 for labels)
#         f.write(struct.pack(">I", 2049))
#         # Write number of labels
#         f.write(struct.pack(">I", num_samples))
#         # Write labels (as bytes)
#         f.write(labels.astype(np.uint8).tobytes())

# # Convert images & labels to NumPy arrays
# train_images = train_dataset.data.numpy()
# train_labels = train_dataset.targets.numpy()
# test_images = test_dataset.data.numpy()
# test_labels = test_dataset.targets.numpy()

# # Save images and labels in IDX format
# save_idx_images(os.path.join(DATA_DIR, "train-images-idx3-ubyte"), train_images)
# save_idx_labels(os.path.join(DATA_DIR, "train-labels-idx1-ubyte"), train_labels)
# save_idx_images(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte"), test_images)
# save_idx_labels(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte"), test_labels)

print("MNIST dataset saved in ~/data/MNIST/raw in original format!")