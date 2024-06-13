import os
import numpy as np
from PIL import Image
from typing import List
import struct

def load_images(path: str, limit=None) -> List[np.ndarray]:
    images = []
    image_count = 0
    for file in os.listdir(path):
        if image_count == limit:
            break
        image = Image.open(os.path.join(path, file)).resize((256, 256)).convert('RGB')
        image = np.array(image)
        images.append(image)
        image_count += 1
    return images

def save_images(images: List[np.ndarray], path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    for i, image in enumerate(images):
        image = Image.fromarray(image)
        image.save(os.path.join(path, f'{i+1}.png'))

def preprocess_images(images: List[np.ndarray]) -> np.ndarray:
    images = np.array(images)
    images = images / 255.0
    images = images.reshape(images.shape[0], -1)
    return images

def postprocess_images(images: np.ndarray) -> List[np.ndarray]:
    images = images.reshape(images.shape[0], 256, 256, 3)
    images = images * 255.0
    images = images.astype(np.uint8)
    return images


def readMNIST(path_prefix='./data/MNIST/', verbose=True):
    def read_idx(filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.int8).reshape(shape)
        
    train_images = read_idx(path_prefix+'train-images-idx3-ubyte/train-images-idx3-ubyte')
    test_images = read_idx(path_prefix+'t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')

    train_labels = read_idx(path_prefix+'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_labels = read_idx(path_prefix+'t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    if verbose:
        print(f'Training images shape: {train_images.shape}')
        print(f'Training labels shape: {train_labels.shape}')
        print(f'Test images shape: {test_images.shape}')
        print(f'Test labels shape: {test_labels.shape}')
        
    return train_images, train_labels, test_images, test_labels

def one_hot_encode(y, n_classes):
    return np.eye(n_classes)[y]
