import logging
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from data_preprocessing.MFCC_CNN import MfccCnn
from models.model_nn import MyModel

if __name__ == '__main__':
    # mfcc_cnn = MfccCnn(n_mfcc=32)
    # mfcc_cnn.get_mfcc_from_signal()
    # mfcc_cnn.targets()

    with open('mfcc_data.npy', 'rb') as f:
        mfcc_data = np.load(f)
    with open('y_data.npy', 'rb') as f:
        y_data = np.load(f)
    X = np.array(mfcc_data)
    y = np.array(y_data)
    (unique, counts) = np.unique(y, return_counts=True)
    print(unique, counts)
    indexes = np.where(y == 1)
    X = np.delete(X, indexes, 0)
    y = np.delete(y, indexes, 0)
    y = np.where(y < 1, y, y - 1)

    input_shape = (mfcc_data.shape[1], mfcc_data.shape[2], 1)

    model = MyModel(7)
    logging.info('Training network..')
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    history = model.fit(X_train, y_train, batch_size=128, epochs=15, validation_data=(x_valid, y_valid))
    print(model.summary())
    score = model.evaluate(X_test, y_test, verbose=2)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(15)

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
    # targets = [0, 1, 2, 3, 4, 5, 6]
    labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # predict probabilities for test set
    y_pred = model.predict(X_test, verbose=0)
    # predict crisp classes for test set
    # yhat_classes = model.predict_classes(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Metryki
    # accuracy: (tp + tn) / (p + n)
    print(classification_report(y_test, y_pred_classes, target_names=labels))

    matrix = confusion_matrix(y_test, y_pred_classes)

    plt.subplots(figsize=(10, 10))
    sns.heatmap(matrix, annot=True, annot_kws={"size": 10}, fmt='.1f', xticklabels=labels, yticklabels=labels)
    logging.info('done')
