import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Notes
# Two Nx10x10 networks as inputs into 10x100x3 2nd layer of network
# ReLU activation
# Uses bias vectors
# Adam optimizer

# Parameters
k1 = 0.001 # Optimizer learning rate
k2 = 0.1 # Regularizer weight
optimizer_fn = tf.keras.optimizers.Adam(learning_rate=k1)
loss_fn = tf.keras.losses.MeanAbsolutePercentageError()
weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=42)
bias_init = tf.keras.initializers.Zeros()

#TODO: 
# 1) Figure out how to separate hidden_11 and hidden_22 --- DONE
# 2) Figure out how to add batchNormalization
# 3) Figure out how to add regularizers --- DONE
# 4) Figure out how to add a loss function, optimizer --- DONE
# 5) Make the layer sizes adaptable to numIn, numOut parameters from process_data.py --- Can't do it
# Example for how to make disconnected layers: output1 = keras.layers.Dense(2)(hidden2[:,0:1])

# Create the model
input = tf.keras.layers.Input(shape=(9,))
# print("input = ", tf.shape(input))
hidden_11 = tf.keras.layers.Dense(20, activation="relu", 
                                  kernel_initializer=weight_init, bias_initializer=bias_init,
                                  kernel_regularizer=tf.keras.regularizers.L2(k2), 
                                  bias_regularizer=tf.keras.regularizers.L2(k2))(input[:,:3])
# batchNorm_11 = tf.keras.layers.BatchNormalization(inputs=hidden_11)
hidden_12 = tf.keras.layers.Dense(20, activation="relu", 
                                  kernel_initializer=weight_init, bias_initializer=bias_init,
                                  kernel_regularizer=tf.keras.regularizers.L2(k2), 
                                  bias_regularizer=tf.keras.regularizers.L2(k2))(input[:,3:])
# print(input[:3])
# print(input[3:])
# print("hidden_11, hidden_12 = ", tf.shape(hidden_11), tf.shape(hidden_12))
# batchNorm_12 = tf.keras.layers.BatchNormalization(inputs=hidden_12)
hidden_21 = tf.keras.layers.Dense(20, activation="relu", 
                                  kernel_initializer=weight_init, bias_initializer=bias_init,
                                  kernel_regularizer=tf.keras.regularizers.L2(k2), 
                                  bias_regularizer=tf.keras.regularizers.L2(k2))(hidden_11)
# batchNorm_21 = tf.keras.layers.BatchNormalization(inputs=hidden_21)
hidden_22 = tf.keras.layers.Dense(20, activation="relu", 
                                  kernel_initializer=weight_init, bias_initializer=bias_init,
                                  kernel_regularizer=tf.keras.regularizers.L2(k2), 
                                  bias_regularizer=tf.keras.regularizers.L2(k2))(hidden_12)
# print("hidden_21, hidden_22 = ", tf.shape(hidden_21), tf.shape(hidden_22))
# batchNorm_22 = tf.keras.layers.BatchNormalization(inputs=hidden_22)
hidden_3 = tf.keras.layers.Dense(20, activation="relu", 
                                 kernel_initializer=weight_init, bias_initializer=bias_init,
                                 kernel_regularizer=tf.keras.regularizers.L2(k2), 
                                 bias_regularizer=tf.keras.regularizers.L2(k2))(tf.keras.layers.concatenate([hidden_21, hidden_22]))
# print("hidden_3 = ", tf.shape(hidden_3))
# batchNorm_3 = tf.keras.layers.BatchNormalization(inputs=hidden_3)
hidden_4 = tf.keras.layers.Dense(100, activation="relu", 
                                 kernel_initializer=weight_init, bias_initializer=bias_init,
                                 kernel_regularizer=tf.keras.regularizers.L2(k2), 
                                 bias_regularizer=tf.keras.regularizers.L2(k2))(hidden_3)
# print("hidden_4 = ", tf.shape(hidden_4))
# batchNorm_4 = tf.keras.layers.BatchNormalization(inputs=hidden_4)
output = tf.keras.layers.Dense(3, activation="relu", 
                               kernel_initializer=weight_init, bias_initializer=bias_init,
                               kernel_regularizer=tf.keras.regularizers.L2(k2), 
                               bias_regularizer=tf.keras.regularizers.L2(k2))(hidden_4)
# print("output = ", tf.shape(output))
biggerHidden = tf.keras.models.Model(input, output)
biggerHidden.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=[tf.keras.metrics.MeanSquaredError(),
                                                                  tf.keras.metrics.MeanAbsoluteError(),
                                                                  tf.keras.metrics.MeanAbsolutePercentageError()])

# Old Models
# model_1 = tf.keras.Sequential([
#     tf.keras.layers.Dense(7,activation='relu'),
#     tf.keras.layers.Dense(100,activation='relu'),
#     tf.keras.layers.Dense(3,activation='relu')
# ])
# model_2 = tf.keras.Sequential([
#     tf.keras.layers.Dense(7,activation='relu'),
#     tf.keras.layers.Dense(100,activation='relu'),
#     tf.keras.layers.Dense(100,activation='relu'),
#     tf.keras.layers.Dense(3,activation='relu')
# ])

# model_1.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=['accuracy'])


# Useful Pandas Commands
# print(dataframe.head())
# dataframe.info()
# print(dataframe.isnull().sum())
# x.info()
# y.info()