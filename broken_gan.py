import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.contrib import rnn
from tensorflow.keras.layers import LSTM, Dense, Input, Conv3D, MaxPooling3D, Flatten, RepeatVector, Lambda
from tensorflow.keras.models import Model, Sequential
import time

from process_data import read_small

num_bars = 4
num_timesteps = 48
#num_timesteps=5

num_pitches = 84
num_tracks = 2

batch_size = 100
hidden_size = 256

def combine_melody_accomp(tensors):
    stacked = tf.stack((tensors[0], tensors[1]), axis=-1)
    reshaped = tf.reshape(stacked,
            (batch_size, num_bars, num_timesteps, num_pitches, num_tracks))
    print("Generator output shape:")
    print(reshaped.shape)
    return reshaped

def build_generator():
    """
    Create two-output hierarchical LSTM model to generate two tracks.
    Diagram of arch:
                            melody_inter -> melody_out
    input -> melody_lstm <
                            accomp_lstm -> accomp_inter -> accomp_out
    """
    # Input is [batch_size x num_pitches]
    inp = Input(shape=(num_pitches,), batch_size=batch_size)

    repeated = RepeatVector(num_bars * num_timesteps)(inp)
    melody_lstm = LSTM(num_pitches, return_sequences=True)(repeated)
    accomp_lstm = LSTM(num_pitches, return_sequences=True)(melody_lstm)
    out = Lambda(combine_melody_accomp)([melody_lstm, accomp_lstm])

    return Model(inp, out)

def build_discriminator():
    """
    Convolutional discriminator
    """
    inp = Input(
            shape=(num_bars, num_timesteps, num_pitches, num_tracks),
            batch_size=batch_size)

    conv1 = Conv3D(
            32, kernel_size=(1, 2, 4),
            activation='relu')(inp)

    conv2 = Conv3D(
            32, kernel_size=(2, 2, 2),
            activation='relu')(conv1)

    pool1 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)

    flat = Flatten()(pool1)
    dense = Dense(256, activation='relu')(flat)
    softmax = Dense(1, activation='softmax')(dense)

    return Model(inp, softmax)


class GAN(object):
    def __init__(self):
        self.optimizer = Adam(lr=0.1)

        # Create generator
        self.G = build_generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        noise = Input(shape=(num_pitches,), batch_size=batch_size)
        generated_seq = self.G(noise)
        print("Generated sequence:")
        print(generated_seq.shape)

        # Create discriminator
        self.D = build_discriminator()
        self.D.compile(loss='binary_crossentropy',
                optimizer=self.optimizer, metrics=['accuracy'])

        self.D.trainable = False
        validity = self.D(generated_seq)

        # Create combined model for training generator
        self.stacked_gd = Model(noise, validity)
        self.stacked_gd.compile(
                loss='binary_crossentropy', optimizer=self.optimizer)


    def train(self, x_train, epochs=5, batch=batch_size):
        print("Training...")
        for e in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, x_train.shape[0], batch_size//2)
            real = x_train[idx]
            noise = tf.convert_to_tensor(np.random.uniform(
                    -1,1, size=(batch_size, num_pitches)).astype(np.float32))
            generated = self.G(noise)[:batch_size//2]

            x_combined = tf.concat((real, generated), axis=0)
            y_combined = tf.concat((
                tf.ones((batch_size//2,1)),
                tf.zeros((batch_size//2,1))), axis=0)
            permute = np.random.permutation(list(range(batch_size)))
            d_loss = self.D.train_on_batch(x_combined, y_combined)

            # Train generator
            noise = tf.convert_to_tensor(np.random.uniform(
                    -1,1, size=(batch_size, num_pitches)))
            g_loss = self.stacked_gd.train_on_batch(
                    noise,
                    np.ones((np.int64(batch_size), 1)))

            print("Disc loss: "+str(d_loss))
            print("Gen loss: "+str(g_loss))
            

if __name__ == '__main__':
    small = read_small().astype(np.float32)
    gan = GAN()
    gan.train(small)
