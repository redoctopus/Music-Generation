# -*- coding: utf-8 -*-


import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties


def main(argv):
    note_labels = ["C", 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if len(argv) < 3:
        return
    notes = np.load(argv[1])
    if notes.shape[-1] == 5:
        notes = notes[:,:,:,:,[2,4]]
    
    notes = notes[:20]
    notes = notes.reshape((20,4,48,84,2)) # 48 for lakh, 8 for ours
    notes[notes < 0] = 0
    notes[notes > 0] = 1

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
    note_histogram = np.zeros(6)
    note_tr = np.zeros((6,6))
    music = notes.reshape(-1,84,2).transpose(2,0,1)
    lengths = {}
    lengths2 = set()
    lastlastlength = -1
    lastlength = -1
    for track in music:
        for i in range(0, music.shape[1]-5, 4):
            indices = np.nonzero(track[i])[0] % 12
            ls = []
            #print(indices)
            for l in lengths.keys():
                if l not in indices:
                    lastlastlength = lastlength
                    if l == 1:
                        note_histogram[0] += 1 #8th
                        lastlength = 0
                    elif l == 2:
                        note_histogram[1] += 1 #quarter
                        lastlength = 1
                    elif l == 3:
                        note_histogram[2] += 1 #quarter dot
                        lastlength = 2
                    elif l == 4:
                        note_histogram[3] += 1 #half
                        lastlength = 3
                    elif l <= 6:
                        note_histogram[4] += 1 #half dot
                        lastlength = 4
                    elif l <= 8:
                        note_histogram[5] += 1
                        lastlength = 5

                    ls.append(l)
                    lengths2.add(l)
                else:
                    lengths[l] += 1
            if lastlastlength > 0 and lastlength > 0:
                note_tr[lastlastlength][lastlength] += 1
            for l in ls:
                del lengths[l]
            for l in indices:
                if l not in lengths:
                    lengths[l] = 1
            histogram[indices] += 1
            next_indices = np.nonzero(track[i+1])[0] % 12
            i1 = indices#np.asarray(list(set(indices).difference(next_indices)), dtype='int64')
            #print(indices, next_indices)
            i2 = next_indices#np.asarray(list(set(next_indices).difference(indices)), dtype='int64')
            #print(i1, i2)
            transitions[i1.repeat(len(i2)),np.tile(i2, len(i1))] += 1
        indices = np.nonzero(track[-1])[0] % 12
        histogram[indices] += 1
    plt.bar(range(12), histogram)
    plt.yticks([])
    plt.xticks(range(12), note_labels)
    plt.savefig(argv[2] + '_pch.png')
    print('pitch histogram saved')
    plt.clf()
    #fig, ax = plt.subplots()
    transitions_norm = transitions / max(1, transitions.max())
    plt.xticks(range(12), note_labels)
    plt.yticks(range(12), note_labels)
    plt.imshow(transitions_norm, cmap='jet')
    plt.colorbar()
    plt.savefig(argv[2] + '_pctm.png')
    print('pitch transitions saved')
    plt.clf()

    nc = len(lengths2)
    prop = FontProperties()
    prop.set_file('FreeSerif.ttf')

    print('note count {}'.format(nc))
    fig, ax = plt.subplots()
    symbolsx = [u"\U0001D160", u"\U0001D15F", u"\U0001D15F\U0001D16D", u"\U0001D15E", u"\U0001D15E\U0001D16D", u"\U0001D15D"]
    plt.bar(range(6), note_histogram)
    ax.set_xticks(range(len(symbolsx)))
    ax.set_xticklabels(symbolsx, fontsize=500, fontproperties=prop)
    ax.set_yticks([])
    plt.savefig(argv[2] + '_nlh.png')
    plt.clf()
    print('note histogram saved')

    transitions_norm = note_tr / max(1, note_tr.max())
    plt.xticks(range(6), symbolsx, size=500, fontproperties=prop)
    plt.yticks(range(6), symbolsx, size=500, fontproperties=prop)
    plt.imshow(transitions_norm, cmap='jet')
    plt.colorbar()
    plt.savefig(argv[2] + '_nctm.png')
    plt.clf()
    print('note transitions saved')


if __name__ == '__main__':
    main(sys.argv)
