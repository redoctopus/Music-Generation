import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.keras.layers import LSTM, Dense, Input, Lambda
from tensorflow.keras.models import Model
import time

from process_data import read_small

num_bars = 4
num_timesteps = 8
#num_timesteps = 48
num_pitches = 84
num_tracks = 2

batch_size = 1000
epochs = 50
seq_len = 8
hidden_size = 256

def combine_melody_accomp(tensors, batch_size=batch_size):
    """Helper function to combine melody and accomp tensors."""
    stacked = tf.stack((tensors[0], tensors[1]), axis=-1)
    reshaped = tf.reshape(stacked,
            (batch_size, num_pitches, num_tracks))
    return reshaped # (batch_size x num_pitches x num_tracks)

def build_generator():
    """
    Create two-output hierarchical LSTM model to generate two tracks.
    Diagram of arch:
                            melody_inter -> melody_out
    input -> melody_lstm <
                            accomp_lstm -> accomp_inter -> accomp_out
    """
    # Input is [batch_size x num_pitches]
    #inp = Input(shape=(seq_len, num_pitches), batch_size=batch_size)
    inp = Input(shape=(seq_len, num_pitches), batch_size=batch_size)

    melody_lstm = LSTM(
            num_pitches, return_sequences=True, stateful=True)(inp)
    accomp_lstm = LSTM(
            num_pitches, return_sequences=True, stateful=True)(melody_lstm)

    melody_inter = LSTM(32)(melody_lstm)
    accomp_inter = LSTM(32)(accomp_lstm)

    melody_out = Dense(num_pitches, activation='softmax')(melody_inter)
    accomp_out = Dense(num_pitches, activation='softmax')(accomp_inter)

    out = Lambda(combine_melody_accomp)([melody_out, accomp_out])

    return Model(inp, out)

def run_generator(model, s):
    """
    Runs the generator for num_bars * num_timesteps steps, and returns 
    the full sequence of generated notes.
    """
    notes = []

    model.reset_states()
    
    for i in range(num_bars * num_timesteps - seq_len):
        preds = model.predict(s, steps=1)

        notes.append(preds)
        melody_preds = tf.reshape(preds[:,:,0], (batch_size, 1, num_pitches))
        s = tf.concat((s[:,1:], melody_preds), axis=1)

    notes = np.stack(notes, axis=1)
    return notes
    

if __name__ == '__main__':
    data = np.load('full_data_sub.npy')
    print("Data shape:")
    print(data.shape)
    data = np.reshape(
            data, (-1, num_bars*num_timesteps, num_pitches, num_tracks))
    print("Data shape (timesteps combined):")
    print(data.shape)
    train_size = data.shape[0]

    model = build_generator()
    model.compile(loss='mse', optimizer='rmsprop')

    # Train model
    for e in range(epochs):
        print("Epoch "+str(e))
        perm = np.random.permutation(train_size)
        data = data[perm]

        for batch in range(0, train_size, batch_size):
            a = batch
            b = batch + batch_size
            if b >= train_size: break

            print("Starting "+str(a)+" to "+str(b))
            model.reset_states()
            # For each batch, train on predicting next time step
            for t in range(num_bars * num_timesteps - seq_len - 1):
                x = data[a:b, t:t+seq_len, :, 0]
                y = data[a:b, t+seq_len, :]
                loss = model.fit(x, y, batch_size=(b-a))
        
        model.save('results/hierarchical_gen_model.h5')

        # Generate some music and save it
        notes = run_generator(model, data[:batch_size, :seq_len, :, 0])
        with tf.Session().as_default():
            notes = np.round(notes)
            print("melody/accomp")
            print(np.sum(notes[:,:,:,0]))
            print(np.sum(notes[:,:,:,1]))
            notes = np.concatenate(
                    [data[:batch_size, :seq_len, :,:], notes], axis=1)
            np.save('results/notes_'+str(e)+'.npy', notes)
