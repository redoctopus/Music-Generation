import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
import time

from process_data import read_small

num_bars = 4
num_timesteps = 48
num_pitches = 84
num_tracks = 2

batch_size = 10
seq_len = 1
hidden_size = 256

def build_generator():
    """
    Create two-output hierarchical LSTM model to generate two tracks.
    Diagram of arch:
                            melody_inter -> melody_out
    input -> melody_lstm <
                            accomp_lstm -> accomp_inter -> accomp_out
    """
    # Input is [batch_size x seq_len x num_pitches]
    inp = Input(shape=(seq_len, num_pitches), batch_size=batch_size)

    melody_lstm = LSTM(
            hidden_size, return_sequences=True, stateful=True)(inp)
    accomp_lstm = LSTM(
            hidden_size, return_sequences=True, stateful=True)(melody_lstm)

    melody_inter = LSTM(32)(melody_lstm)
    accomp_inter = LSTM(32)(accomp_lstm)

    melody_out = Dense(num_pitches, activation='softmax')(melody_inter)
    accomp_out = Dense(num_pitches, activation='softmax')(accomp_inter)
    
    model = Model(inp, [melody_out, accomp_out])
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

def run_generator(model):
    # Create seed values
    s = np.random.uniform(-1,1, size=(batch_size, seq_len, num_pitches))
    melody_notes = []
    accomp_notes = []

    model.reset_states()

    for i in range(num_bars * num_timesteps):
        melody_preds, accomp_preds = model.predict(s)

        # The rounding is funky. We need a way to threshold.
        if len(melody_notes) == 0:
            melody_notes = np.reshape(
                    np.round(melody_preds), (batch_size, 1, -1))
            accomp_notes = np.reshape(
                    np.round(accomp_preds), (batch_size, 1, -1))
        else:
            melody_notes = np.insert(
                    melody_notes, i, np.round(melody_preds), axis=1)
            accomp_notes = np.insert(
                    accomp_notes, i, np.round(accomp_preds), axis=1)

        s = np.insert(s, seq_len, melody_preds, axis=1)[:, 1:]

    print(np.array(melody_notes).shape)
    print(np.array(accomp_notes).shape)
    return melody_notes, accomp_notes
    
    
if __name__ == '__main__':
    model = build_generator()
    notes = run_generator(model)
