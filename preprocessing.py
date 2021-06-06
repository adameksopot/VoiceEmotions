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
        signals.append([signal, i[21], sample_rate])                                 #21th char refers to class (emotion)

    
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

def remove_noise():    
    # to do
    return 1

def play(path):
    ipd.Audio(path)

def zero_crossing_rate(y):
    #działa
    return sum(librosa.zero_crossings(y))

def spectral_centroids(y, sr):
    #w sumie to nie wiem co z tym XD
    return librosa.feature.spectral_centroid(y, sr=sr)[0]

def spectral_roll_off(y, sr):
    #z tym też nie wiem co xd
    S, phase = librosa.magphase(librosa.stft(y))
    return(librosa.feature.spectral_rolloff(S=S, sr=sr))

def MFCC(y, sr):
    #to to wgl jest dziwne
    return librosa.feature.mfcc(y=y, sr=sr)

def energy():
    # to do 
    return 

def spectral_flux(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    return onset_env

def spectral_entropy():
    #to do
    return

def chroma_features(y, sr):
    return librosa.feature.chroma_stft(y=y, sr=sr)

def pitch(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    return pitches
    
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

#play example file
dataset = 0
file = 0
play(paths[dataset][file])

zero_crossing = []
spectral_centr = []
mfcc = []
ener = []
spec_roll_off = []
spec_flux = []
spec_entropy = []
chroma = []
pit = []
for i in data:
    mfcc.append(MFCC(i[0], i[2]))
    zero_crossing.append(zero_crossing_rate(i[0]))
    spectral_centr.append(spectral_centroids(i[0], i[2]))
    #ener.append()
    spec_roll_off.append(spectral_roll_off(i[0], i[2]))
    spec_flux.append(spectral_flux(i[0], i[2]))
    #spec_entropy.append()
    chroma.append(chroma_features(i[0], i[2]))
    pit.append(pitch(i[0], i[2]))
    
    
    

