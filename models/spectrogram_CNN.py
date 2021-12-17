import numpy as np
import tensorflow as tf
from keras import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
with open('spectrograms.npy', 'rb') as f:
    X = np.load(f)
with open('target.npy', 'rb') as f:
    y = np.load(f)

X = np.array(X)
y = np.array(y)

(unique, counts) = np.unique(y, return_counts=True)
print(unique, counts) 

#remove class 1
indexes = np.where(y == 1)
X = np.delete(X, indexes, 0)
y = np.delete(y, indexes, 0)
y = np.where(y<1, y, y-1)

input_shape = (X.shape[1], X.shape[2], 1)

no_of_classes = len(np.unique(y, return_counts=True)[0])

model = Sequential([
    tf.keras.Input(shape=input_shape),                   
    tf.keras.layers.Conv2D(48, kernel_size=2, activation='relu', 
                            padding='SAME'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='SAME'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='SAME'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),
   
    tf.keras.layers.Flatten(),
           
    #tf.keras.layers.BatchNormalization(axis=1),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(no_of_classes, activation='softmax') 
    ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


batch_size=32
epochs=15
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))
tf.keras.models.save_model(model, "model_spectogram")
print(model.summary())



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



y_pred = model.predict(X_test)
targets = [0, 1, 2, 3, 4, 5, 6]
labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

y_pred=np.argmax(y_pred,axis=1)
print(classification_report(y_test, y_pred, target_names=labels))

matrix = confusion_matrix(y_test, y_pred)

plt.subplots(figsize=(10,10))  
sn.heatmap(matrix, annot = True, annot_kws={"size": 10}, fmt = '.1f', xticklabels = labels, yticklabels = labels)
