import sys
sys.path.append('/home/raab/raab_code/thunderfish/thunderfish')
import numpy as np
import matplotlib.pyplot as plt
import dataloader as dl
import audioio as ai
import powerspectrum as ps
import harmonicgroups as hg
import config_tools as ct
from IPython import embed
import pickle
import glob
import time

# function to speperate one fish into two when there are to many nans in between.

# funtion to analyse the distribution of EODfs (glaetten)

# duratin period of fishes being present and overlap.
# if present together --> frequency difference ?

# pickel drop when data is long enought...

# function to load and combine the data...

def pickle_save(fishes):
    file = open('%0.f.p' % (len(glob.glob("*.p"))+1), "wb")
    pickle.dump(fishes, file)
    # embed()

def sort_fishes(all_fundamentals):
    fishes = [[ 0. ]]
    last_fish_fundamentals = [ 0. ]

    # loop throught lists of fundamentals detected in sequenced PSDs.
    for list in range(len(all_fundamentals)):
        # if this list is empty add np.nan to every deteted fish list.
        if len(all_fundamentals[list]) == 0:
                for fish in range(len(fishes)):
                    fishes[fish].append(np.nan)
        # when list is not empty...
        else:
            # ...loop trought every fundamental for each list
            for idx in range(len(all_fundamentals[list])):
                # calculate the difference to all last detected fish fundamentals
                diff = abs(np.asarray(last_fish_fundamentals) - all_fundamentals[list][idx])
                # find the fish where the frequency fits bests (th = 1Hz)
                if diff[np.argsort(diff)[0]] < 1 and diff[np.argsort(diff)[0]] > -1:
                    # add the frequency to the fish array and update the last_fish_fundamentals list
                    fishes[np.argsort(diff)[0]].append(all_fundamentals[list][idx])
                    last_fish_fundamentals[np.argsort(diff)[0]] = all_fundamentals[list][idx]
                # if its a new fish create a list of nans with the frequency in the end (list has same length as other
                # lists.) and add frequency to last_fish_fundamentals
                else:
                    fishes.append([np.nan for i in range(list)])
                    fishes[-1].append(all_fundamentals[list][idx])
                    last_fish_fundamentals.append(all_fundamentals[list][idx])

        # wirte an np.nan for every fish that is not detected in this window!
        for fish in range(len(fishes)):
            if len(fishes[fish]) != list +1:
                fishes[fish].append(np.nan)

    # reshape everything to arrays
    for fish in range(len(fishes)):
        fishes[fish] = np.asarray(fishes[fish])
    # remove first fish because it has beeen used for the first comparison !
    fishes.pop(0)

    return fishes

def main(audio_file):
    cfg = ct.get_config_dict()

    # load file
    # data, samplrate, unit = dl.load_data(audio_file)    #### THIS IS WORKING !!!

    # with dl.open_data(audio_file, 0, 60.0, 10.0) as data:     #### THIS IS SUPRESSING THE ERRORS
    #     samplrate = data.samplerate

    data = dl.open_data(audio_file, 0, 60.0, 10.0)    ### THIS BRINGS ERRORS
    samplrate = data.samplerate

    # fig, ax = plt.subplots()
    idx = 0
    all_fundamentals = []
    t0 = time.time()
    while idx < int((len(data)-8*samplrate) / samplrate):

        tmp_data = data[idx*samplrate : (idx+8)*samplrate]
        psd_data = ps.multi_resolution_psd(tmp_data, samplrate)
        fishlist = hg.harmonic_groups(psd_data[1], psd_data[0], cfg)[0]

        if not fishlist == []:
            fundamentals = hg.extract_fundamental_freqs(fishlist)
            all_fundamentals.append(fundamentals)
            # ax.plot(np.ones(len(fundamentals)) * idx, fundamentals, 'k.')
            # plt.draw()
            # plt.pause(0.001)
        else:
            all_fundamentals.append(np.array([]))

        # print idx
        if idx % 1800 < 0.1 and not idx == 0.0:
            print('%.1f sec; Processing 30 min took %.2f sec.' % (idx, time.time() - t0))
            t0 = time.time()
            # print idx
            fishes = sort_fishes(all_fundamentals)
            # pickle save function
            pickle_save(fishes)
            all_fundamentals = []

        idx += 0.1
    fishes = sort_fishes(all_fundamentals)

if __name__ == '__main__':
    print('TRACK ALL THE FISHES')
    audio_file = sys.argv[1]
    main(audio_file)