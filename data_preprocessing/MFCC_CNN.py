import glob

import numpy as np
import librosa
import librosa.display


class MfccCnn:
    def __init__(self, n_mfcc):
        self.mfcc = []
        self.y = []
        self.n_mfcc = n_mfcc
        self.paths = [glob.glob('data1/*/*.wav'), glob.glob('data2/*/*.wav'), glob.glob('data3/*.wav')]
        self.targets1 = {'n': 1, 'h': 3, 'sa': 4, 'a': 5, 'f': 6, 'd': 7, 'su': 8}
        self.targets2 = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry',
                6: 'fear', 7: 'disgust', 8: 'ps'}
        self.data = self.read_signal()


    def get_key(self, my_dict, val):
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

    def read_signal(self):
        """
        Read signals with annotations
        :param str paths: paths to records
        :param str targets1: dictionary to code emotions from dataset 2
        :param str targets1: dictionary to code emotions from dataset 3

        :return: list signals: [signal, emotion, sample_rate]
        """
        signals = []
        for i in self.paths[0]:
            signal, sample_rate = librosa.load(i)
            signals.append([signal, i[22], sample_rate])  # 21th char refers to class (emotion)

        for i in self.paths[1]:
            signal, sample_rate = librosa.load(i)
            if i[9] == 's':  # class sa or su
                signals.append([signal, self.targets1[i[9:11]], sample_rate])
                continue
            signals.append([signal, self.targets1[i[9]], sample_rate])  # 9 th char refers to class (emotion)

        for i in self.paths[2]:
            # Extract Raw Audio from Wav File
            signal, sample_rate = librosa.load(i)
            signals.append(
                [signal, self.get_key(self.targets2, i[i.rfind('_') + 1:-4]), sample_rate])  # 21th char refers to class (emotion)
        return signals

    def MFCC(self, y, sr):
        mfcc = librosa.feature.mfcc(y=y, n_mfcc=self.n_mfcc, sr=sr)
        return mfcc  # or mfcc_features

    def get_mfcc_from_signal(self):
        for i in self.data:
            self.mfcc.append(self.MFCC(i[0], i[2]))
        self.change_dim()
        self.save_mfcc_to_file('mfcc_data')

    def save_mfcc_to_file(self, filename):
        np.save('{0}.npy'.format(filename), self.mfcc)

    def save_targets_to_file(self, filename):
        np.save('{0}.npy'.format(filename), self.y)

    def targets(self):
        for i in self.data:
            self.y.append(int(i[1]) - 1)
        self.save_targets_to_file('y_data')

    def change_dim(self):
        minimum = 200
        for i in self.mfcc:
            if i.shape[1] < minimum:
                minimum = i.shape[1]

        for i in range(len(self.mfcc)):
            self.mfcc[i] = self.mfcc[i][:, :minimum]
            self.mfcc[i] = np.expand_dims(self.mfcc[i], axis=2)

    # def delete_class(self):
    #     # remove class 1
    #     indexes = np.where(y == 1)
    #     X = np.delete(X, indexes, 0)
    #     y = np.delete(y, indexes, 0)
    #     y = np.where(y < 1, y, y - 1)

