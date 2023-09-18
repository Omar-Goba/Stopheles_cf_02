from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
from keras.layers import *
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
import os
from keras.models import Model
import sys
from sklearn.preprocessing import StandardScaler 
sys.path.append('src')
#from ..data.cooking import init_tensor


"""
I merged , codewise the cooking.py file with this one, since 
I couldnt import the function init_tensor() (tried several techniques my laptop resisted lol)

The training process has horrendous accuracy of 0% which is waiting to be adjusted
"""



def get_tensor(arr: np.array,window_size: int = 7) -> tuple[np.array, np.array]:

    # Create input sequences and corresponding target values
    sequences = []
    targets = []

    for i in range(len(arr) - window_size):
        sequence = arr[i:i+window_size, 1:]  
        target = arr[i+window_size, 0]  

        sequences.append(sequence)
        targets.append(target)

    # Convert lists to NumPy arrays
    X = np.array(sequences)
    y = np.array(targets)  
    return X,y

def init_model(M: int, p: list[int],input_shape: tuple) -> keras.Model:
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
    model = Sequential()
    model.add(InputLayer(input_shape))
    for i in range(M-1):
      model.add(LSTM(p[i], return_sequences=True))
      model.add(Dropout(0.2))
    model.add(LSTM(16, return_sequences=False))
    model.add(Dropout(0.2))
 
    model.add(Dense(8,'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,'linear'))
    model.add(Dropout(0.2))


    return model

def train_model(model: keras.Model, tr_X: np.ndarray, tr_y: np.ndarray, epochs: int = 150, batch_size: int = 32) -> keras.Model:
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
    model.compile(optimizer=Adam(learning_rate = 0.001), loss='mean_squared_error', metrics=['accuracy'])

    # Train the model
    model.fit(tr_X, tr_y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model


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
     # Evaluate the model on the testing data
    loss, accuracy = model.evaluate(ts_X, ts_y)

    return loss, accuracy


def main():
    tr = pd.read_csv("./dbs/cooked/tr.csv")
    ts = pd.read_csv("./dbs/cooked/ts.csv")
    tr = tr.to_numpy()
    ts = ts.to_numpy()
    scaler = StandardScaler()
    

    tr= scaler.fit_transform(tr)
    tr_X, tr_y = get_tensor(tr,7)
    ts_X, ts_y = get_tensor(ts,7)
   
   
    input_shape = tr_X.shape[1:]  # (7, 74)
    model = init_model(M=4, p=[128, 64, 32], input_shape=input_shape)
    model.summary()
    train_model(model, tr_X, tr_y, epochs=150, batch_size= 32)
    loss, accuracy = test_model(model,ts_X, ts_y)
    print("The loss is:" , loss)
    print("The accuracy is:", accuracy)
    
    return 0


if (__name__ == "__main__"):    main()