import numpy as np
import time

large_file = '../musegan/data/train_x_lpd_5_phr.npz'
small_file = 'small_data.npy'

def process_original_data(save_small=False):
    # shape = [n, #bars, #timesteps/bar, #pitches, #tracks]

    with np.load(large_file) as loaded:
        shape = loaded['shape']
        data = np.zeros(shape=shape).astype(bool)

        # Create matrix
        stime = time.time()
        data[[x for x in loaded['nonzero']]] = True
        print("Time taken to load: %s" % (time.time()-stime))

        # Cut to two tracks
        stime = time.time()
        data = data[:, :, :, :, 1::3]

        np.save('full_data.npy', data)

        if save_small:
            np.save(small_file, data[:100])

def read_small():
    filepath = 'small_data.npy'
    data = np.load(filepath)
    print(data.shape)
    return data

def subsample():
    data = np.load('full_data.npy')
    data = data[:,:,0::6]
    print(data.shape)

    np.save('full_data_sub.npy', data)
    np.save('small_data_sub.npy', data[:100])
        

if __name__ == '__main__':
    process_original_data(save_small=True)
    subsample()
