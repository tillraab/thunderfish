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
from scipy.interpolate import spline

def plot_smooth_eodf(clean_fishes, ax, bin_width=0.05):

    for fish in range(len(clean_fishes)):
        tmp_f, tmp_ax = plt.subplots()
        n, bins =  tmp_ax.hist(clean_fishes[fish], bins=np.arange(min(clean_fishes[fish]), max(clean_fishes[fish]) + bin_width, bin_width))[:2]
        plt.close(tmp_f)
        # np.delete(bins, -1)
        bins += bin_width
        np.delete(bins, -1)
        # smooth it
        smooth_n = np.zeros(len(n))

        # ToDo: glaettungsfactor als variable
        # laette +- 10 ....
        for i in range(len(n)):
            if i == 0:
                smooth_n[i] = np.mean([0, 0, n[i], n[i+1], n[i+2]])
            elif i == 1:
                smooth_n[i] = np.mean([0, n[i-1], n[i], n[i+1], n[i+2]])
            elif i == len(n)-2:
                smooth_n[i] = np.mean([n[i-2], n[i-1], n[i], n[i+1], 0])
            elif i == len(n)-1:
                smooth_n[i] = np.mean([n[i-2], n[i-1], n[i], 0, 0])
            else:
                smooth_n[i] = np.mean([n[i-1], n[i], n[i+1]])

        ax.plot(bins[:-1]+bin_width, smooth_n)
    # embed()
    # quit()


def plot_fundamentals(fishes, ax, ystart):
    for fish in range(len(fishes)):
        ax.plot((np.arange(len(fishes[fish])) * 1.024 + ystart) / 60., fishes[fish], '.k', markersize=2)
    ax.set_xlabel('time [min]')
    ax.set_ylabel('frequency [Hz]')
    ystart += 900
    return ystart

def fish_freq_distibution(fishes):
    clean_fishes = []
    for fish in range(len(fishes)):
        if len(fishes[fish][~np.isnan(fishes[fish])]) >= 50:
            tmp_fish = fishes[fish][~np.isnan(fishes[fish])]
            clean_fishes.append(np.asarray(tmp_fish))

    # embed()
    # quit()
    return clean_fishes

def combine_fishes(all_fishes):
    # all_fishes[snippet][fish array]
    fishes = all_fishes[0]

    for snippet in range(1, len(all_fishes)):
        for fish in range(len(all_fishes[snippet])):
            fish_last_f = np.array([fishes[i][-1] for i in range(len(fishes))])
            diff = abs(fish_last_f - all_fishes[snippet][fish][0])
            if diff[np.argsort(diff)[0]] < 3.:
                fishes[np.argsort(diff)[0]] = np.concatenate((fishes[np.argsort(diff)[0]], all_fishes[snippet][fish]))
            else:
                fishes.append(all_fishes[snippet][fish])

    return fishes


def main(filepath):

    file_counts = len(glob.glob(filepath + '/*.p'))

    fig, ax = plt.subplots()
    # fig2, ax2 = plt.subplots()
    ystart = 0.
    all_clean_fishes =[]
    for eNum, f_no in enumerate(np.arange(file_counts)+1):
        f = open(filepath + '/%0.f.p' % f_no, 'rb')
        fishes = pickle.load(f)
        if fishes == []:
            continue
        else:
            ystart = plot_fundamentals(fishes, ax, ystart)
            plt.draw()
            plt.pause(0.001)

            clean_fishes = fish_freq_distibution(fishes)
            all_clean_fishes.append(clean_fishes)

            f.close()
            del f
            plt.draw()
            plt.pause(0.001)

            # if eNum > 3:
            #     quit()
    sorted_fishes = combine_fishes(all_clean_fishes)
    embed()
    quit()

    plt.show()

if __name__ == '__main__':
    filepath = sys.argv[1]
    main(filepath)