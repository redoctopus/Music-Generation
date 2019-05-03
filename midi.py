import numpy as np
import pypianoroll
import sys

def save_pianoroll(filename, pianoroll, programs, is_drums, tempo,
                   beat_resolution, lowest_pitch):
    """Saves a batched pianoroll array to a npz file."""
    if not np.issubdtype(pianoroll.dtype, np.bool_):
        raise TypeError("Input pianoroll array must have a boolean dtype.")
    if pianoroll.ndim != 5:
        raise ValueError("Input pianoroll array must have 5 dimensions.")
    if pianoroll.shape[-1] != len(programs):
        raise ValueError("Length of `programs` does not match the number of "
                         "tracks for the input array.")
    if pianoroll.shape[-1] != len(is_drums):
        raise ValueError("Length of `is_drums` does not match the number of "
                         "tracks for the input array.")

    reshaped = pianoroll.reshape(
        -1, pianoroll.shape[1] * pianoroll.shape[2], pianoroll.shape[3],
        pianoroll.shape[4])

    # Pad to the correct pitch range and add silence between phrases
    to_pad_pitch_high = 128 - lowest_pitch - pianoroll.shape[3]
    padded = np.pad(
        reshaped, ((0, 0), (0, pianoroll.shape[2]),
                   (lowest_pitch, to_pad_pitch_high), (0, 0)), 'constant')

    # Reshape the batched pianoroll array to a single pianoroll array
    pianoroll_ = padded.reshape(-1, padded.shape[2], padded.shape[3])

    # Create the tracks
    tracks = []
    for idx in range(pianoroll_.shape[2]):
        tracks.append(pypianoroll.Track(
            pianoroll_[..., idx], programs[idx], is_drums[idx]))

    # Create and save the multitrack
    multitrack = pypianoroll.Multitrack(
        tracks=tracks, tempo=tempo, beat_resolution=beat_resolution)
    multitrack.write(filename)

def save_midi(infile, outfile):
    array = np.asarray(list(map(bool, np.load(infile).flatten()))).reshape((-1,4,8,84,2))
    #array = np.asarray(list(map(bool, np.load(infile).flatten()))).reshape((-1,4,48,84,2))
    save_pianoroll(
        outfile, # filename
        array, # notes
        [0, 48], # programs
        [False, False], #is_drums
        100, #tempo
        2, #beat_resolution
        #12, #beat_resolution
        24) #lowest pitch

def main(argv):
    if len(argv) < 3:
        print("Run 'python midi.py notes.npy out.mid'")
        return
    save_midi(argv[1], argv[2])

if __name__ == '__main__':
    main(sys.argv)
