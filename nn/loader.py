from typing import Tuple, Dict
import os
import requests
import gzip
import numpy as np


class DataLoader():
    """ Batch Data Loader that shuffles the data at every epoch
    Args:
        data (np.ndarray): 2D Data array
        labels (np.ndarray): 1D class/label array
        batch_size (int): Batch size
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, batch_size: int):
        self.batch_size = batch_size
        self.data = self.preprocess(data)
        self.labels = labels
        self.index = None

    def __iter__(self) -> "DataLoader":
        """ Shuffle the data and reset the index
        Returns:
            DataLoader: self object
        """
        self.shuffle()
        self.index = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Return a batch of data and label starting at <self.index>. Also increment <self.index>.
        Raises:
            StopIteration: If builtin "next" function is called when the data is fully passed 
        Returns:
            Tuple[np.ndarray, np.ndarray]: Batch of data (B, D) and label (B) arrays
        """
        if self.index is None or self.index >= len(self.data):
            raise StopIteration

        start_index = self.index
        end_index = self.index + self.batch_size
        self.index = end_index

        batch_data = self.data[start_index:end_index]
        batch_labels = self.labels[start_index:end_index]

        return batch_data, batch_labels

    def shuffle(self) -> None:
        """ Shuffle the data
        """
        permutation = np.random.permutation(len(self.data))
        self.data = self.data[permutation]
        self.labels = self.labels[permutation]

    @staticmethod
    def preprocess(data: np.ndarray) -> np.array:
        """ Preprocess the data
        Args:
            data (np.ndarray): data array
        Returns:
            np.array: Float data array with values ranging from 0 to 1 
        """
        return data.astype(np.float32) / 255


class FashionMnistDataset():

    train_labels_url: str  = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
    train_images_url: str  = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
    test_labels_url: str = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
    test_images_url: str = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"

    @classmethod
    def download_and_load(cls, url: str, name: str, kind: str, offset: int) -> np.ndarray:
        """ Load or download the data if it does not exist

        Args:
            url (str): Download url
            name (str): name of the file
            kind (str): Train or test
            offset (int): Byte offset

        Returns:
            np.ndarray: _description_
        """
        os.makedirs("data", exist_ok=True)
        file_path = f"data/{kind}_{name}"
        if not os.path.exists(file_path):
            response = requests.get(url)
            with open(file_path, "wb") as f:
                f.write(response.content)
        with gzip.open(file_path, "rb") as lbpath:
            return np.frombuffer(lbpath.read(), dtype=np.uint8, offset=offset)

    @classmethod
    def load(cls) -> Dict[str, np.ndarray]:
        """ Load the data

        Returns:
            Dict[str, np.ndarray]: Train data and labels and test data and labels
        """
        train_labels = cls.download_and_load(cls.train_labels_url, "train", "labels", 8)
        train_data = cls.download_and_load(cls.train_images_url, "train", "images",
                                    16).reshape(len(train_labels), 784)
        test_labels = cls.download_and_load(cls.test_labels_url, "test", "labels", 8)
        test_data = cls.download_and_load(cls.test_images_url, "test", "images", 16).reshape(len(test_labels), 784)
        return dict(train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)


