import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import glob
import soundfile
import random
import simpleaudio as sa
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
# The letters 'a', 'd', 'f', 'h', 'n', 'sa' and 'su' represent 'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sad' and 'surprise' 
# (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral)

def get_key(my_dict, val):
    for key, value in my_dict.items():
         if val == value:
             return key
    return "key doesn't exist"

def readSignal(paths, targets1, targets2):
    signals = []
    for i in paths[0]:
        spf = wave.open(i, "r")
        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, "Int16")
        signals.append([signal, i[21]])                                 #21th char refers to class (emotion)
    
    
    for i in paths[1]:
        spf = wave.open(i, "r")
        # Extract Raw Audio from Wav File
        signal = spf.readframes(-1)
        signal = np.fromstring(signal, "Int16")
        if i[9] == 's':                                        #class sa or su 
            signals.append([signal, targets1[i[9:11]]])                  
            continue
        signals.append([signal, targets1[i[9]]])                         #9th char refers to class (emotion)
        
    for i in paths[2]:
        # Extract Raw Audio from Wav File
        signal, samplerate = soundfile.read(i)
        signals.append([signal, get_key(targets2, i[i.rfind('_')+1:-4])])                              #21th char refers to class (emotion)
        
    
    return signals

def plotExample(signal, x):
    plt.figure(1)
    plt.title("Signal Wave "+str(signal[1])+'record: '+str(x))
    plt.plot(signal[0])
    plt.show()

def removeSilence(signal):
    
    return signal

def play(signal, sample_rate, num_channels = 1, bytes_per_sample = 2):
    play_obj = sa.play_buffer(signal, num_channels, bytes_per_sample, sample_rate)

targets1 = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
targets2 = {'n':1, 'h':3, 'sa':4, 'a':5, 'f':6, 'd':7, 'su': 8}
targets3 = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8: 'ps'}

paths = []
paths.append(glob.glob('data1/*/*.wav'))
paths.append(glob.glob('data2/*/*.wav'))
paths.append(glob.glob('data3/*.wav'))

data = readSignal(paths, targets2, targets3)

for i in range(0, 10):
    x = random.randint(0, len(data))
    plotExample(data[x], x)

#play example file
sample_rate1 = 48000    #dataset 1 - records from 0 to  1439
sample_rate2 = 44100    #dataset 2 - records from 1440 to 1919  
sample_rate3 = 24414    #dataset 3 - records from 1920 to  4719

play(data[496][0], sample_rate1)

