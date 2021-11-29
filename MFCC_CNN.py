import numpy as np
import glob
import librosa
import librosa.display
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_key(my_dict, val):
    """
    get dictionary key from value
    :param dictionary my_dict: dictionary
    :param any value: value
    :param str targets1: dictionary to code emotions from dataset 3
    
    :return: list signals: [signal, emotion, sample_rate]
    """
    for key, value in my_dict.items():
         if val == value:
             return key
    return "key doesn't exist"

def read_signal(paths, targets1, targets2):
    """
    Read signals with annotations 
    :param str paths: paths to records
    :param str targets1: dictionary to code emotions from dataset 2
    :param str targets1: dictionary to code emotions from dataset 3
    
    :return: list signals: [signal, emotion, sample_rate]
    """
    signals = []
    for i in paths[0]:
        signal, sample_rate = librosa.load(i)
        signals.append([signal, i[22], sample_rate])                                 #21th char refers to class (emotion)

    
    for i in paths[1]:
        signal, sample_rate = librosa.load(i)
        if i[9] == 's':                                                              #class sa or su 
            signals.append([signal, targets1[i[9:11]], sample_rate])                  
            continue
        signals.append([signal, targets1[i[9]], sample_rate])                        #9th char refers to class (emotion)
        
    for i in paths[2]:
        # Extract Raw Audio from Wav File
        signal, sample_rate = librosa.load(i)
        signals.append([signal, get_key(targets2, i[i.rfind('_')+1:-4]),sample_rate])                              #21th char refers to class (emotion)
    return signals

def MFCC(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, n_mfcc=n_mfcc, sr=sr)
    return mfcc #or mfcc_features
    
targets1 = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry',
            6: 'fearful', 7: 'disgust', 8: 'surprised'}
targets2 = {'n':1, 'h':3, 'sa':4, 'a':5, 'f':6, 'd':7, 'su': 8}
targets3 = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry',
            6:'fear', 7:'disgust', 8: 'ps'}

paths = []
paths.append(glob.glob('data1/*/*.wav'))
paths.append(glob.glob('data2/*/*.wav'))
paths.append(glob.glob('data3/*.wav'))

data = read_signal(paths, targets2, targets3)           

mfcc = []
for i in data:
    mfcc.append(MFCC(i[0], i[2], 13))


y=[]
for i in data:
   y.append(int(i[1])-1) 

minimum = 200
for i in mfcc:
   if i.shape[1] < minimum:
       minimum = i.shape[1]

for i in range(len(mfcc)):
    mfcc[i] = mfcc[i][:,:minimum]       
    mfcc[i] = np.expand_dims(mfcc[i], axis=2)
    mfcc[i] = np.resize(mfcc[i], (32,55))
    
X = np.array(mfcc)    
y = np.array(y)


    
input_shape = (32, minimum, 1)       
model_to_train = tf.keras.applications.ResNet50(
    include_top=True, weights=None, input_tensor=None,
    input_shape=input_shape,
    pooling=None, classes=8,
    classifier_activation='sigmoid'
)

model_to_train.compile(optimizer = 'Adam', loss='categorical_crossentropy',metrics='Accuracy')
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

history = model_to_train.fit(X_train, y_train, batch_size=32, epochs = 10, validation_data=(x_valid, y_valid))