import os
import jax
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

def save_images(images: List[jax.numpy.array], path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i, image in enumerate(images):
        image = np.array(image)
        # print('image array:',image)
        image = np.round(image)
        image = Image.fromarray(image.astype(np.uint8))
        image.save(os.path.join(path, f'{i}.png'))

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

from array import array

class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)

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
