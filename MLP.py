import pandas as pd
import numpy as np
import tensorflow as tf
import os
import pickle
import seaborn as sn
import keras
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

data = pd.read_csv('mfcc22_df.csv', index_col=False)
labels = data.iloc[:, [-1]]
labels = labels - 1

data = data.drop(labels.columns, axis=1)
data.describe()

data = data.to_numpy()
labels = labels.to_numpy()
(unique, counts) = np.unique(labels, return_counts=True)
print(unique, counts)

indexes = np.where(labels == 1)
data = np.delete(data, indexes, 0)
labels = np.delete(labels, indexes, 0)
labels = np.where(labels < 1, labels, labels - 1)

(unique, counts) = np.unique(labels, return_counts=True)
print(unique, counts)

"""##Standarization"""

data += 848.919070  # adding abs min value to whole data just to elimiate negative values
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)

"""##Splitting data

"""

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print("Number of training samples:", X_train.shape[0])
print("Number of testing samples:", X_test.shape[0])
print("Number of val samples:", X_val.shape[0])
print("Number of features:", X_train.shape[1])

"""#KERAS

"""

num_classes = 7

keras_model = tf.keras.Sequential([

    tf.keras.Input(shape=(22,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

keras_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
keras_model.summary()

batch_size = 256
epochs = 120
learning_rate = 'adaptive'

history = keras_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])

test_results = keras_model.evaluate(X_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

predict_x = keras_model.predict(X_test)
classes_x = np.argmax(predict_x, axis=1)

matrix = confusion_matrix(y_test, classes_x)

x_axis_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

plt.subplots(figsize=(10, 10))
sn.heatmap(matrix, annot=True, annot_kws={"size": 10}, fmt='.1f', xticklabels=x_axis_labels, yticklabels=x_axis_labels)

print(classification_report(y_test, classes_x, target_names=x_axis_labels))

"""## Saving model"""

if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(keras_model, open("result/TensorflowMlpClassifier.model", "wb"))
