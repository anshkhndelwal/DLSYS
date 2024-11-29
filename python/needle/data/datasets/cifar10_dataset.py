import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms=transforms
        if train:
            files = []
            for i in range(1, 6):
              files.append(f'data_batch_{i}')
        else:
            files = ['test_batch']
        X, y = [], []
        for d_file in files:
          with open(os.path.join(base_folder, d_file), 'rb') as f:
            data = pickle.load(f, encoding = 'bytes')
            X.append(data[b'data'])
            y.append(data[b'labels'])
        X = (np.concatenate(X, axis = 0).astype(np.float32))/255.
        self.X = X.reshape((-1, 3, 32, 32))
        self.y = np.concatenate(y, axis = None).astype(np.float32)
        
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if self.transforms:
            transformed_images = []
            for img in self.X[index]:
                transformed_images.append(self.apply_transforms(img))
            img = np.array(transformed_images, dtype=np.float32)
        else:
            img = self.X[index]
        return img, self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)        
        ### END YOUR SOLUTION
