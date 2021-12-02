import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import glob
import soundfile
import random
import simpleaudio as sa
import librosa
import IPython.display as ipd
import librosa.display
import pandas as pd
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

def plot_example(signal, x, sr):
    plt.title('emotion: '+str(x))
    librosa.display.waveplot(signal, sr=sr)
    plt.show()


def MFCC(y, sr, n_mfcc=13):
    mfcc = np.mean(librosa.feature.mfcc(y=y, n_mfcc=n_mfcc, sr=sr), axis=1) #n_mfcc 13 or 22 it will depend on kind of emotion
    #first derivative
    # delta1 = librosa.feature.delta(mfcc)
    # #second derivative
    # delta2 = librosa.feature.delta(mfcc, order = 2)
    # mfcc_features = np.concatenate((mfcc, delta1, delta2))
    # print(mfcc_features.shape)
    # #calculating mean and variance 
    # mfcc_mean = mfcc.mean(axis=1) #we can add it to features, we will see
    # mfcc_variance = mfcc.var(axis=1) #we can add it to features, we will see
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

data = read_signal(paths, targets2, targets3)           #long time

for i in range(0, 10):
    x = random.randint(0, len(data))
    plot_example(data[x][0], data[x][1], data[x][2])


mfcc13 = []
for i in data:
    mfcc13.append(MFCC(i[0], i[2], 13))

mfcc22 = []
for i in data:
    mfcc22.append(MFCC(i[0], i[2], 22))

y=[]
for i in data:
   y.append(i[1]) 
   
mfcc13_df = pd.DataFrame(np.c_[np.array(mfcc13), np.array(y)])
mfcc13_df.to_csv('mfcc13_df.csv', index=False)  
mfcc22_df = pd.DataFrame(np.c_[np.array(mfcc22), np.array(y)])
mfcc22_df.to_csv('mfcc22_df.csv', index=False)


# signal, sample_rate = librosa.load('data1/Actor_01/03-01-01-01-01-01-01.wav')
# mfcc = np.mean(librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sample_rate), axis=1)
    
    

