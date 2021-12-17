import glob
import numpy as np
import librosa

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
        signals.append([signal, i[22], sample_rate])  # 21th char refers to class (emotion)

    for i in paths[1]:
        signal, sample_rate = librosa.load(i)
        if i[9] == 's':  # class sa or su
            signals.append([signal, targets1[i[9:11]], sample_rate])
            continue
        signals.append([signal, targets1[i[9]], sample_rate])  # 9 th char refers to class (emotion)

    for i in paths[2]:
        # Extract Raw Audio from Wav File
        signal, sample_rate = librosa.load(i)
        signals.append(
            [signal, get_key(targets2, i[i.rfind('_') + 1:-4]), sample_rate])  # 21th char refers to class (emotion)
    return signals


def melspectrogram(y, sr):
    return librosa.feature.melspectrogram(y=y, sr=sr)

targets1 = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry',
            6: 'fearful', 7: 'disgust', 8: 'surprised'}
targets2 = {'n': 1, 'h': 3, 'sa': 4, 'a': 5, 'f': 6, 'd': 7, 'su': 8}
targets3 = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry',
            6: 'fear', 7: 'disgust', 8: 'ps'}

paths = [glob.glob('data1/*/*.wav'), glob.glob('data2/*/*.wav'), glob.glob('data3/*.wav')]

data = read_signal(paths, targets2, targets3)

melspectrograms = []
for i in data:
    melspectrograms.append(melspectrogram(i[0], i[2]))

y = []
for i in data:
    y.append(int(i[1]) - 1)

minimum = 200
for i in melspectrograms:
    if i.shape[1] < minimum:
        minimum = i.shape[1]

for i in range(len(melspectrograms)):
    melspectrograms[i] = melspectrograms[i][:, :minimum]
    melspectrograms[i] = np.expand_dims(melspectrograms[i], axis=2)

# saving mfcc
saved = np.save('models/spectrograms.npy', melspectrograms)

