import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import keras
from keras import optimizers
#
# with open('test.npy', 'rb') as f:
#     mefa = np.load(f)
# with open('target.npy', 'rb') as f:
#     y1 = np.load(f)


class MyModel(tf.keras.Model):
    def __init__(self, classes):
        super(MyModel, self).__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), activation='relu')
        self.conv2d_2 = tf.keras.layers.Conv2D(48, kernel_size=(2, 2), activation='relu')
        self.conv2d_3 = tf.keras.layers.Conv2D(120, kernel_size=(2, 2), activation='relu')
        self.maxpool2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.4)
        self.dense2 = tf.keras.layers.Dense(classes, activation='softmax')

    def call(self, inputs):
        my_model = self.conv2d_1(inputs)
        my_model = self.conv2d_2(my_model)
        my_model = self.conv2d_3(my_model)
        my_model = self.maxpool2d(my_model)
        my_model = self.flatten(my_model)
        my_model = self.dense1(my_model)
        my_model = self.drop(my_model)
        my_model = self.dense2(my_model)
        # return the constructed model
        return my_model

    def save_model(self):
        pass

    def import_model(self):
        pass

# X = np.array(mefa)
# y = np.array(y1)
#
# input_shape = (mefa.shape[1], mefa.shape[2], 1)

#
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='Adam',
#               metrics=['accuracy'])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
#
# history = model.fit(X_train, y_train, batch_size=128, epochs=15, validation_data=(x_valid, y_valid))
#
# score = model.evaluate(X_test, y_test, verbose=2)
# print(score)
