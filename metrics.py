import sys
import numpy as np
import matplotlib.pyplot as plt

def main(argv):
    if len(argv) < 2:
        return
    notes = np.load(argv[1])
    if len(notes.shape) == 4:
        notes = notes.reshape((1, *notes.shape))

    # Pitch metrics
    pitches = np.amax(notes.transpose(0,1,2,4,3).flatten().reshape(-1,84), axis=0)
    pc = np.sum(pitches)
    print("pitch count: {}".format(pc))
    ranges = np.nonzero(pitches)[0]
    pr = ranges[-1] - ranges[0]
    print("pitch range: {}".format(pr))

    # Pitch and note metrics
    histogram = np.zeros(12)
    transitions = np.zeros((12,12))
    
    note_histogram = np.zeros(48)

    music = notes.reshape(-1,84,2).transpose(2,0,1)
    lengths = {}
    for track in music:
        for i in range(music.shape[1]-1):
            indices = np.nonzero(track[i])[0] % 12
            ls = []
            for l in lengths.keys():
                if l not in indices:
                    note_histogram[lengths[l]] += 1
                    ls.append(l)
                else:
                    lengths[l] += 1
            for l in ls:
                del lengths[l]
            for l in indices:
                if l not in lengths:
                    lengths[l] = 1
            histogram[indices] += 1
            next_indices = np.nonzero(track[i+1])[0] % 12
            transitions[indices.repeat(len(next_indices)),np.tile(next_indices, len(indices))] += 1
        indices = np.nonzero(track[-1])[0] % 12
        histogram[indices] += 1
    plt.bar(range(12), histogram)
    plt.yticks([])
    plt.xticks(range(12), range(1,13))
    plt.savefig(argv[1] + 'pitch_histogram.png')
    print('pitch histogram saved')
    plt.clf()
    fig, ax = plt.subplots()
    ax.imshow(transitions)
    plt.savefig(argv[1] + 'pitch_transition.png')
    print('pitch transitions saved')
    plt.clf()
    pr = np.nonzero(note_histogram)[0].size
    print('note count {}'.format(pr))
    plt.bar(range(48), note_histogram)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(argv[1] + 'note_histogram.png')
    print('note histogram saved')


if __name__ == '__main__':
    main(sys.argv)