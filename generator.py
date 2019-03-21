import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.keras.layers import LSTM, Dense
import time

from process_data import read_small

num_bars = 4
num_timesteps = 48
num_pitches = 84
num_tracks = 2

batch_size = 10
seq_len = 3
hidden_size = 256

def build_generator():
    model = tf.keras.Sequential([
        LSTM(
            hidden_size, return_sequences=True,
            input_shape=(seq_len, num_pitches)),
        LSTM(
            hidden_size, return_sequences=True),
        LSTM(32),
        Dense(num_pitches, activation='softmax')])

    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model
        
    
if __name__ == '__main__':
    s = np.random.uniform(-1,1, size=(10, seq_len, num_pitches))
    model = build_generator()
    notes = []

    for i in range(num_bars * num_timesteps):
        preds = model.predict(s)
        # The rounding is funky. We need a way to threshold.
        if len(notes) == 0:
            notes = np.reshape(np.round(preds), (batch_size, 1, -1))
        else:
            notes = np.insert(notes, i, np.round(preds), axis=1)

        print(notes.shape)

        print(s.shape)
        print(preds.shape)

        s = np.insert(s, seq_len, preds, axis=1)[:, 1:]

    print(np.array(notes).shape)

