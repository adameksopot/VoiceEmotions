import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


with open('test.npy', 'rb') as f:
    mefa = np.load(f)
with open('target.npy', 'rb') as f:
    y1 = np.load(f)

X = np.array(mefa)
y = np.array(y1)

input_shape = (mefa.shape[1], mefa.shape[2], 1)

vgg16_model = tf.keras.applications.vgg16.VGG16(
    include_top=True,
    weights=None,
    input_shape=input_shape,
)
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
# for layer in model.layers:
#     layer.trainable = False
model.add(Dense(units=8, activation='sigmoid'))
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics='Accuracy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

history = model.fit(X_train, y_train, batch_size=64, epochs=15, validation_data=(x_valid, y_valid))
