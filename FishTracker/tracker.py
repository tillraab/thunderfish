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

    pickle.dump(fishes, open('%0.f.p' % (len(glob.glob("*.p"))+1), "wb"))
    # embed()

def sort_fishes(all_fundamentals):
    fishes = [[all_fundamentals[0][i]] for i in range(len(all_fundamentals[0]))]
    last_fish_fundamentals = [fishes[i][-1] for i in range(len(fishes))]

    for list in range(1, len(all_fundamentals)):
        for idx in range(len(all_fundamentals[list])):
            diff = abs(np.asarray(last_fish_fundamentals) - all_fundamentals[list][idx])
            if diff[np.argsort(diff)[0]] < 1 and diff[np.argsort(diff)[0]] > -1:
                fishes[np.argsort(diff)[0]].append(all_fundamentals[list][idx])
                last_fish_fundamentals[np.argsort(diff)[0]] = all_fundamentals[list][idx]
            else:
                fishes.append([np.nan for i in range(list)])
                fishes[-1].append(all_fundamentals[list][idx])
                last_fish_fundamentals.append(all_fundamentals[list][idx])

        # wirte an np.nan for every fish that is not detected in this window!
        for fish in range(len(fishes)):
            if len(fishes[fish]) != list +1:
                fishes[fish].append(np.nan)
        # need method to assing fish with same frequency as already known to a new fish when it has been gone for to long

    # reshape everything to arrays
    for fish in range(len(fishes)):
        fishes[fish] = np.asarray(fishes[fish])
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

    while idx < int((len(data)-8*samplrate) / samplrate):
        if idx > 1200:
            quit() # quit at 20 minutes !!!
        t0 = time.time()
        tmp_data = data[idx*samplrate : (idx+8)*samplrate]
        psd_data = ps.multi_resolution_psd(tmp_data, samplrate)
        fishlist = hg.harmonic_groups(psd_data[1], psd_data[0], cfg)[0]

        if not fishlist == []:
            fundamentals = hg.extract_fundamental_freqs(fishlist)
            all_fundamentals.append(fundamentals)
            # ax.plot(np.ones(len(fundamentals)) * idx, fundamentals, 'k.')
            # plt.draw()
            # plt.pause(0.001)
        # print idx
        if idx % 60 < 0.1 and not idx == 0.0:
            fishes = sort_fishes(all_fundamentals)
            # pickle save function
            pickle_save(fishes)

            all_fundamentals = []
            # embed()
            # quit()
 # fishes[0][~np.isnan(fishes[0])]    gets the elements that are not nan !!!
        idx += 0.1

        print idx, "%.2f sec" % (time.time() - t0)
    fishes = sort_fishes(all_fundamentals)
    # plt.show()
    # embed()
    # quit()

if __name__ == '__main__':
    print('TRACK ALL THE FISHES')
    audio_file = sys.argv[1]
    main(audio_file)