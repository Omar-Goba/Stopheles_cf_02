from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


def get_tensors(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        loads all the tensors from the path
        args:
            path: path to the tensors
        returns:
            tuple of tensors
                - tr_X
                - tr_y
                - ts_X
                - ts_y
    """
    ...


def init_model(M: int, p: list[int]) -> keras.Model:
    """
        initializes the LSTM Model with M layers
        the architecture is:
            - M LSTM layers with p[i] neurons
            = M dropout layers with 0.2 dropout rate
            - 1 dense layer with 8 neuron
            - 1 dense layer with 1 neuron
        use the functional api from keras
        args:
            M: number of layers
            p: list of number of neurons per layer
                len(p) == M
        returns:
            keras model
    """
    ...


def train_model(model: keras.Model, tr_X: np.ndarray, tr_y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> keras.Model:
    """
        trains the model
        args:
            model: keras model
            tr_X: training data
            tr_y: training labels
            epochs: number of epochs
            batch_size: batch size
        returns:
            keras model
    """
    ...


def test_model(model: keras.Model, ts_X: np.ndarray, ts_y: np.ndarray) -> tuple[float, float]:
    """
        tests the model
        args:
            model: keras model
            ts_X: testing data
            ts_y: testing labels
        returns:
            tuple of metrics
                - loss
                - accuracy
    """
    ...


def main():
    ...


if (__name__ == "__main__"):    main()
