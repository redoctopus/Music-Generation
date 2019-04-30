import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.keras.layers import LSTM, Dense, Input, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras import backend as K

from pypianoroll import Multitrack
import time
import argparse

from process_data import read_small

num_bars = 4
num_timesteps = 48
num_pitches = 84
num_tracks = 2

batch_size = 5
epochs = 5
seq_len = 4
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

def run_generator(model, s=[]):
    """
    Runs the generator for num_bars * num_timesteps steps, and returns 
    the full sequence of generated notes.
    """
    # Create seed values
    if len(s) == 0:
        s = tf.convert_to_tensor(
                np.random.uniform(-1,1, size=(batch_size, seq_len, num_pitches))
                .astype(np.float32))
    else:
        s = tf.convert_to_tensor(s)
    notes = []

    model.reset_states()

    for i in range(num_bars * num_timesteps):
        preds = model(s)

        notes.append(preds)
        melody_preds = tf.reshape(preds[:,:,0], (batch_size, 1, num_pitches))
        s = tf.concat((s, melody_preds), axis=1)[:, 1:]

    notes = tf.stack(notes, axis=1)
    print("Final notes:")
    print(notes.shape)
    return notes
    

if __name__ == '__main__':
    # initiate the parser
    parser = argparse.ArgumentParser()

    # add args
    parser.add_argument("--train", "-r", help="train model", action="store_true")
    parser.add_argument("--generate", "-g", help="generate from trained model", action="store_true")

    # read arguments from the command line
    args = parser.parse_args()
    
    data = read_small()
    #data = np.load('full_data.npy')
    print("Data shape:")
    print(data.shape)
    data = np.reshape(
            data, (-1, num_bars*num_timesteps, num_pitches, num_tracks))
    print("Data shape (timesteps combined):")
    print(data.shape)
    train_size = data.shape[0]

    if args.train:
        model = build_generator()
        model.compile(loss='mse', optimizer='rmsprop')

        # Train model
        for e in range(epochs):
            perm = np.random.permutation(train_size)
            data = data[perm]

            for batch in range(0, train_size, batch_size):
                a = batch
                b = batch + batch_size
                model.reset_states()
                # For each batch, train on predicting next time step
                for t in range(num_bars * num_timesteps - seq_len - 1):
                    x = data[a:b, t:t+seq_len, :, 0]
                    y = data[a:b, t+seq_len, :]
                    loss = model.fit(x, y, batch_size=batch_size)

            model.save('hierarchical_gen_model.h5')
            
    elif args.generate:
        model = load_model('hierarchical_gen_model.h5')
        '''
        tb_path = "logs/"
        graph = K.get_session().graph
        writer = tf.summary.FileWriter(logdir=tb_path, graph=graph)
        '''
        # Error when trying to use data for generation
        #notes = run_generator(model, s=data[:batch_size, :seq_len, :, 0])
        notes = run_generator(model, s=[])
        # Saving does not work ("TypeError: can't pickle _thread.RLock objects")
        np.save('notes.npy', notes)
