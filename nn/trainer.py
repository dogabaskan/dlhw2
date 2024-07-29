from typing import Any, Tuple
import numpy as np

from autograd.array import Array
from nn.loader import DataLoader
from nn.logger import Logger
from nn.layers import Module
from autograd.functions import nll_with_logits_loss, softmax
from autograd import grad


class Train():
    """ Trainer class that updates the model, evaluates it, and logs the training process. 

    Args:
        model (Module): Neural Network Model
        optimizer (Any): Optimizer that has the parameter of the given model
    """

    def __init__(self, model: Module, optimizer: Any) -> None:
        self.model = model
        self.optimizer = optimizer

    def predict(self, features: Array) -> np.ndarray:
        """ Predict a class

        Args:
            features (Array): Input Array of shape (B, D)

        Returns:
            np.ndarray: Prediction NumPy array of shape (B) # an integer for each feature in the batch
        """
        logits = self.model(features)
        return logits.value.argmax(axis=-1)

    def one_step_update(self, batch_data: np.ndarray, batch_labels: np.ndarray) -> Tuple[float, float]:
        """ Update the model using a single batch. Return loss and accuracy.
        Use ```nll_with_logits_loss``` as the loss function.
        Note that, model outputs must be logits!

        Args:
            batch_data (np.ndarray): NumPy array of shape (B, D)
            batch_labels (np.ndarray): NumPy array of shape (B)

        Returns:
            Tuple[float, float]:
                - Loss of the model on the given batch
                - Avarage accuracy of the model on the given batch
        """
        features = Array(batch_data)
        labels = Array(batch_labels)

        #print("FEATURES", features.shape)
        #print("LABELS", labels.shape)
        
        logits = self.model(features)
        
        #print("LOGITS", logits.shape)
             
        loss_value = nll_with_logits_loss(logits, labels)
        loss_val = loss_value.value.mean()
        

        #print("LOSS VAL", loss_value)
        gradients = grad(loss_value)

        self.optimizer.update(gradients)

        predictions = self.predict(features)
        predictions = softmax(predictions)
        accuracy = self.accuracy(predictions, batch_labels)


        return loss_value, accuracy
    
    def fit(self,
            train_data_loader: DataLoader,
            eval_data_loader: DataLoader,
            epochs: int,
            logger: Logger) -> None:
        """ Main training function
        Args:
            train_data_loader (DataLoader): Data loader with training data
            eval_data_loader (DataLoader): Data loader with evaluation data
            learning_rate (float): Learning rate
            l2_regularization_coeff (float): L2 regularization coefficient
            epochs (int): Number of epochs
            logger (Logger): Logger object for logging accuracies and losses
        """

        for epoch_index in range(epochs):
            self.model.train_mode()
            train_epoch_accuracy_list = []
            train_epoch_loss_list = []

            for iter_index, (train_data, train_label) in enumerate(train_data_loader):

                train_loss, train_accuracy = self.one_step_update(train_data, train_label)

                train_epoch_accuracy_list.append(train_accuracy)
                train_epoch_loss_list.append(train_loss)
                logger.iter_train_accuracy.append(train_accuracy)
                logger.iter_train_loss.append(train_loss)
                logger.log_iteration(epoch_index, iter_index)

            self.model.eval_mode()
            epoch_eval_acc = self.evaluate(eval_loader=eval_data_loader)

            logger.epoch_train_accuracy.append(np.mean(train_epoch_accuracy_list))
            logger.epoch_eval_accuracy.append(epoch_eval_acc)
            logger.log_epoch(epoch_index)

    def evaluate(self, eval_loader: DataLoader) -> float:
        """ Evaluate the data using eval loader.

        Args:
            eval_loader (DataLoader): Data loader that contains evaluation data

        Returns:
            np.ndarray: Average accuracy
        """
        acc_list = []
        for iter_index, (eval_data, eval_label) in enumerate(eval_loader):
            prediction = self.predict(Array(eval_data))
            eval_accuracy = self.accuracy(prediction, eval_label)
            acc_list.append(eval_accuracy)
        return np.mean(acc_list).item()

    def test(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """ Evaluate the model on test dataset.

        Args:
            test_loader (DataLoader): Test data loader

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - predictions of shape (B)
                - True labels of shape (B) 
        """
        labels = []
        predictions = []
        self.model.eval_mode()
        for iter_index, (test_data, test_label) in enumerate(test_loader):
            prediction = self.predict(Array(test_data))
            predictions.append(prediction)
            labels.append(test_label)
        return np.concatenate(predictions), np.concatenate(labels)
            

    @staticmethod
    def accuracy(prediction: np.ndarray, label: np.ndarray) -> np.float32:
        """ Calculate mean accuracy
        Args:
            prediction (np.ndarray): Prediction array of shape (B)
            label (np.ndarray): Ground truth array of shape (B)
        Returns:
            np.float32: Average accuracy
        """
        return (prediction == label).mean()

    @staticmethod
    def confusion_matrix(predictions: np.ndarray, label: np.ndarray) -> np.ndarray:
        """ Calculate confusion matrix
        Args:
            predictions (np.ndarray): Prediction array of shape (B)
            label (np.ndarray): Ground truth array of shape (B)
        Returns:
            np.ndarray: Confusion matrix of shape (C, C)
        """
        num_classes = np.max(label) + 1
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        for p, l in zip(predictions, label):
            cm[l, p] += 1
        return cm