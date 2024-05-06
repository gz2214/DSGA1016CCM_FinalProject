# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:53:07 2024

@author: dinos
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:46:47 2024

@author: dinos
"""

import argparse
import numpy as np
from scipy.special import softmax
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd 
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import importlib

from helper_for_BaselineNN.utility_functions import UtilityBase
from helper_for_BaselineNN.weighting_functions import WeightingBase


def main_trainer(all_util_func, all_weight_func, frac_data):
    data=pd.read_csv("c13k_selections.csv")
    data=data.sample(frac=frac_data, random_state=42)

    util_weight_all=[]
    results=[]
    for util_func in all_util_func:
        for weight_func in all_weight_func:
            util_weight_all.append(f"util_{util_func}_weight_{weight_func}")
            results.append(individual_trainer(util_func, weight_func, data))

    plt.figure(figsize=(10, 6))
    for func, r in zip(util_weight_all, results):
        capped_val_loss = [min(x, 50) for x in r.history['loss']]
        plt.plot(capped_val_loss, label=func)
        
    plt.title('Training Loss of Different Models')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig('plots/training_loss.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    for func, r in zip(util_weight_all, results):
        capped_val_loss = [min(x, 50) for x in r.history['val_loss']]
        plt.plot(capped_val_loss, label=func)
        
    plt.title('Validation Loss of Different Models')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.savefig('plots/validation_loss.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()


def individual_trainer(util_func, weight_func, data):
    utility_fn = getattr(
                importlib.import_module("utility_functions"), util_func
            )()
    weight_fn = getattr(
                importlib.import_module("weighting_functions"), weight_func
            )()

    A=[[[weight_fn(data['pHa'][i]),utility_fn(data['Ha'][i])],[weight_fn(1-data['pHa'][i]),utility_fn(data['La'][i])]] for i in range(14568)]
    A =np.array([i for i in A])
    B=[[[weight_fn(data['pHb'][i]),utility_fn(data['Hb'][i])],[weight_fn(1-data['pHb'][i]),utility_fn(data['Lb'][i])]] for i in range(14568)]
    B =np.array([i for i in B])
    X = np.concatenate((A, B), axis=1)
    Y=np.array(data['bRate'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(4, 2)),  # Input shape is (4, 2) for each data point
        Dense(64, activation='relu'),
        Dense(1)  # Output layer without activation function for regression task
    ])

    learning_rate = 0.01  # Custom learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=32, verbose=1)
    return history


if __name__ == "__main__":
    # python BaselineNN.py --util_func_list IdentityUtil LinearUtil AsymmetricLinearUtil LinearLossAverseUtil PowerLossAverseUtil ExpLossAverseUtil NormExpLossAverseUtil NormLogLossAverseUtil NormPowerLossAverseUtil QuadLossAverseUtil LogLossAverseUtil ExpPowerLossAverseUtil GeneralLinearLossAverseUtil GeneralPowerLossAverseUtil NeuralNetworkUtil
    parser = argparse.ArgumentParser()
    # parser.add_argument('--baseline', help='If the baseline model is chosen', default=True)
    parser.add_argument("--util_func_list", nargs='+', help='A list of utility function to use', default=["IdentityUtil"])
    parser.add_argument('--weight_func_list', nargs='+', help='A list of probability weighting function to use', default=["IdentityPWF"])
    parser.add_argument('--frac_data', help='fraction of the 13k data will be used', default=1)

    args = parser.parse_args()

    main_trainer(args.util_func_list, args.weight_func_list, args.frac_data)