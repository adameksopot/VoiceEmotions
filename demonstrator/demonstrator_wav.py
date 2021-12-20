import librosa
from tensorflow import keras


class Demonstrator:
    def __init__(self, n_mfcc):
        self.n_mfcc = n_mfcc
        self.signal = None
        self.sample_rate = None
        self.mfcc = None

    def read_signal(self, path):
        self.signal, self.sample_rate = librosa.load(path)

    def get_mfcc(self):
        self.mfcc = librosa.feature.mfcc(y=self.signal, n_mfcc=self.n_mfcc, sr=self.sample_rate)
        return self.mfcc

    def reshape(self):
        pass

    def predict(self):
        model = keras.models.load_model('path/to/location')
        model.predict(self.mfcc)  # reshape mfcc
