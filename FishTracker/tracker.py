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
import matplotlib.mlab as mlab
import pickle
import glob
# import time

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

def spectogram(data, samplerate, fresolution=0.5, detrend=mlab.detrend_none, window=mlab.window_hanning, overlap=0.5,
               pad_to=None, sides='default', scale_by_freq=None):
    nfft = int(np.round(2 ** (np.floor(np.log(samplerate / fresolution) / np.log(2.0)) + 1.0)))
    if nfft < 16:
        nfft = 16
    noverlap = nfft*overlap
    spectrum, freqs, time = mlab.specgram(data, NFFT=nfft, Fs=samplerate, detrend=detrend, window=window,
                                          noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)

    return spectrum, freqs, time

def main(audio_file):
    import time
    cfg = ct.get_config_dict()

    data = dl.open_data(audio_file, 0, 60.0, 10.0)    ### THIS BRINGS ERRORS
    samplrate = data.samplerate

    idx = 0
    all_fundamentals = []
    loop = 1.
    while idx < int((len(data)-900*samplrate) / samplrate):

        tmp_data = data[idx*samplrate : (idx+908)*samplrate]
        print('data loaded ...')

        spectrum, freqs, time = spectogram(tmp_data, samplrate)
        # psd_data = ps.multi_resolution_psd(tmp_data, samplrate)
        print('spectogramm calculated ...')

        for t in range(len(time)-7):
            power = np.mean(spectrum[:, t:t+8], axis=1)

            fishlist = hg.harmonic_groups(freqs, power, cfg)[0]

            if not fishlist == []:
                fundamentals = hg.extract_fundamental_freqs(fishlist)
                all_fundamentals.append(fundamentals)
            else:
                all_fundamentals.append(np.array([]))

        print('Loop %0.f: Processed %0.f minutes ... continue...' % (loop, loop*15))
        loop +=1
        fishes = sort_fishes(all_fundamentals)
        pickle_save(fishes)
        all_fundamentals = []

        idx += 900.

    fishes = sort_fishes(all_fundamentals)
    print('Whole file processed!')

if __name__ == '__main__':
    print('TRACK ALL THE FISHES')
    audio_file = sys.argv[1]
    main(audio_file)