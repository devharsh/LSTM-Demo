#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 13:34:00 2022.

@author: devharsh
"""

import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(None,), dtype="float64")
xem = layers.Embedding(101, 128)(inputs)
xl1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(xem)
xl2 = layers.Bidirectional(layers.LSTM(64))(xl1)
outputs = layers.Dense(1, activation="sigmoid")(xl2)
model = keras.Model(inputs, outputs)
model.summary()

dataframe = read_csv("sonar.csv", header=None)
dataframe *= 100

encoder = LabelEncoder()
encoder.fit(dataframe[60])
dataframe[60] = encoder.transform(dataframe[60])

print(dataframe.head)
print(dataframe.columns)
print(dataframe.dtypes)

dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=64, epochs=32, validation_data=(x_test, y_test)
)

labels = ["loss", "val_loss"]
for lab in labels:
    plt.plot(history.history[lab], label=lab)
plt.legend()
plt.show()
