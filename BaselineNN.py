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


#%%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
#%%
data=pd.read_csv("c13k_selections.csv")

#%%
#for neural network
#input_data = np.concatenate((np.array(data['A']), np.array(data['B'])), axis=1)

A=[[[data['pHa'][i],data['Ha'][i]],[1-data['pHa'][i],data['La'][i]]] for i in range(14568)]
A =np.array([i for i in A])

B=[[[data['pHb'][i],data['Hb'][i]],[1-data['pHb'][i],data['Lb'][i]]] for i in range(14568)]
B =np.array([i for i in B])

X = np.concatenate((A, B), axis=1)


Y=np.array(data['bRate'])

#%%
model = Sequential([
    Dense(64, activation='relu', input_shape=(4, 2)),  # Input shape is (4, 2) for each data point
    Dense(64, activation='relu'),
    Dense(1)  # Output layer without activation function for regression task
])
#%%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#%%

learning_rate = 0.01  # Custom learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32, verbose=1)
#%%
