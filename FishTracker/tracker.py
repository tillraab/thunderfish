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
import time

def sort_fishes(all_fundamentals):
    print 'in func'
    fishes = [[all_fundamentals[0][i]] for i in range(all_fundamentals[0])]
    # print fishes
    embed()
    quit()

    for list in range(1, len(all_fundamentals)):
        print list
        for idx in range(len(all_fundamentals[list])):
            diff = all_fundamentals[list] - all_fundamentals[list-1][idx]
            if np.argsort(diff)[0] < 1:
                print ''


def main(audio_file):
    cfg = ct.get_config_dict()

    # load file
    with dl.open_data(audio_file, 0, 60.0, 10.0) as data:
        samplrate = data.samplerate
    # data, samplrate, unit = dl.load_data(audio_file)

        fig, ax = plt.subplots()

        idx = 0
        all_fundamentals = []

        while idx < int((len(data)-8*samplrate) / samplrate):
            t0 = time.time()
            tmp_data = data[idx*samplrate : (idx+8)*samplrate]
            psd_data = ps.multi_resolution_psd(tmp_data, samplrate)
            fishlist = hg.harmonic_groups(psd_data[1], psd_data[0], cfg)[0]

            if not fishlist == []:
                fundamentals = hg.extract_fundamental_freqs(fishlist)
                all_fundamentals.append(fundamentals)
                ax.plot(np.ones(len(fundamentals)) * idx, fundamentals, 'k.')
                plt.draw()
                plt.pause(0.001)
            print idx % 1

            if idx % 1 < 0.1 and not idx == 0.0:
                sort_fishes(all_fundamentals)

            idx += 0.1
            # print time.time() -t0
        plt.show()

if __name__ == '__main__':
    print('TRACK ALL THE FISHES')
    audio_file = sys.argv[1]
    main(audio_file)