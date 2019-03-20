import numpy as np
import time

def read_large():
    # shape = [n, #bars, #timesteps/bar, #pitches, #tracks]

    filepath = '../musegan/data/train_x_lpd_5_phr.npz'
    with np.load(filepath) as loaded:
        shape = loaded['shape']
        data = np.zeros(shape=shape).astype(bool)
        stime = time.time()
        data[[x for x in loaded['nonzero']]] = True

        print("Time taken: %s" % (time.time()-stime))

        np.save('small_tracks.npy', data[:100])

def read_small():
    filepath = 'small_tracks.npy'
    data = np.load(filepath)
    print(data.shape)
        

if __name__ == '__main__':
    read_small()
