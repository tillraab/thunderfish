import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import embed
import time
import matplotlib.patches as mpatches
import scipy.stats as scp
from scipy.optimize import curve_fit
from tqdm import tqdm

def Q10(datafile, fish_nr_in_rec):
    def Q10_val(t0, t1, f0, f1):
        return (f1 / f0)**(10/(t1 -t0))

    fish_freq_temp = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    temp = [26, 26, 26, 26.3, 26.0, 26.0, 25.8, 25.8, 25.8, 25.4, 25.4, 25.2, 25.2, 25.1, 25.1, 25.0, 25.0, 24.9, 24.9, 24.8, 24.8, 24.9]
    shift = [0, 5760, 12840, 72420, 169440, 204840, 243900, 255360, 270240, 335400, 362400, 418440, 445380, 505200, 543240, 592740, 630660, 678780, 713700, 765540, 802680, 852840]

    for datei_nr in range(len(datafile)):
        fund_v, ident_v, idx_v, times_v, sign_v = loaddata(datafile[datei_nr])
        # times_v += shift[datei_nr]

        for fish_nr in range(len(fish_nr_in_rec)):
            if np.isnan(fish_nr_in_rec[fish_nr][datei_nr]):
                fish_freq_temp[fish_nr].append([np.nan, np.nan])

                continue

            f = fund_v[ident_v == fish_nr_in_rec[fish_nr][datei_nr]]
            t = times_v[ident_v == fish_nr_in_rec[fish_nr][datei_nr]]

            if len(t[t < 1000]) == 0:
                fish_freq_temp[fish_nr].append([np.nan, np.nan])
                continue
            else:
                fish_freq_temp[fish_nr].append([np.median(f[t < 100]), temp[datei_nr]])

    all_Q10 = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for fish in range(len(fish_freq_temp)):
        for i in range(len(fish_freq_temp[fish])):
            for j in np.arange(i+1, len(fish_freq_temp[fish])):
                if np.isnan(fish_freq_temp[fish][i][1]):
                    continue
                if fish_freq_temp[fish][i][1] == fish_freq_temp[fish][j][1]:
                    continue
                Cq10 = Q10_val(fish_freq_temp[fish][i][1], fish_freq_temp[fish][j][1], fish_freq_temp[fish][i][0], fish_freq_temp[fish][j][0])
                all_Q10[fish].append(Cq10)
    nonan_all_Q10 = []
    n = []
    for q in all_Q10:
        nonan_all_Q10.append(np.array(q)[~np.isnan(np.array(q))])
        n.append(len(nonan_all_Q10[-1]))

    nonan_all_Q10.append([])
    nonan_all_Q10.append(np.hstack(nonan_all_Q10))

    fig, ax = plt.subplots(figsize=(20/2.54, 12/2.54))
    ax.boxplot(nonan_all_Q10, sym='')
    for enu, x in enumerate(n):
        ax.text(enu+1, 0.25, '%.0f' % x, ha='center', va='center')
        ax.text(enu+1, np.median(nonan_all_Q10[enu]), '%.2f' % np.median(nonan_all_Q10[enu]), ha='center')

    ax.text(16, np.median(nonan_all_Q10[-1]), '%.2f' % np.median(nonan_all_Q10[-1]), ha = 'center')
    # ax.boxplot(np.hstack(nonan_all_Q10), positions = [16], sym='')
    ax.set_ylim([0, 5])
    # ax.set_xlim([0, 17])
    plt.show()
    # embed()
    # quit()

def efunc(x, tau):
    return np.exp(-x/tau) / tau

def loaddata(datafile):
    print('loading datafile: %s' % datafile)
    fund_v=np.load(datafile+"/fund_v.npy")
    ident_v=np.load(datafile+"/ident_v.npy")
    idx_v=np.load(datafile+"/idx_v.npy")
    times=np.load(datafile+"/times.npy")
    sign_v=np.load(datafile+"/sign_v.npy")
    times_v=times[idx_v]

    return fund_v,ident_v,idx_v,times_v,sign_v

def create_plot(datafile, shift, fish_nr_in_rec, colors, dn_borders, last_time, ax = None):
    print('plotting traces')

    fish_freqs = []
    solo_plot = False
    if ax == None:
        # fig, ax = plt.subplots(facecolor='white', figsize=(20. / 2.54, 10. / 2.54))
        fig = plt.figure(facecolor='white', figsize=(20. / 2.54, 10. / 2.54))
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        solo_plot = True
    for datei_nr in range(len(datafile)):
        fund_v, ident_v, idx_v, times_v, sign_v = loaddata(datafile[datei_nr])
        times_v += shift[datei_nr]

        for fish_nr in range(len(fish_nr_in_rec)):

            if np.isnan(fish_nr_in_rec[fish_nr][datei_nr]):
                continue
            p = sign_v[ident_v == fish_nr_in_rec[fish_nr][datei_nr]][:, 0]
            f = fund_v[ident_v == fish_nr_in_rec[fish_nr][datei_nr]]
            t = times_v[ident_v == fish_nr_in_rec[fish_nr][datei_nr]]
            if datei_nr == len(datafile)-1:
                fish_freqs.append(np.median(f[-100:]))

            ax.plot(t[~np.isnan(p)], f[~np.isnan(p)], color=colors[fish_nr])

    temp = [26, 26, 26, 26.3, 26.0, 26.0, 25.8, 25.8, 25.8, 25.4, 25.4, 25.2, 25.2, 25.1, 25.1, 25.0, 25.0, 24.9,
                    24.9,
                    24.8, 24.8, 24.9, 26.3, 27.0]

    shift = [0, 5760, 12840, 72420, 169440, 204840, 243900, 255360, 270240, 335400, 362400, 418440, 445380, 505200,
                     543240, 592740, 630660, 678780, 713700, 765540, 802680, 852840, 1559460, 2161500]


    if solo_plot:
        for ns, ne in zip(dn_borders[::2], dn_borders[1::2]):
            ax.fill_between([ns, ne], [650, 650], [970, 970], color='#888888')
        ax.set_yticks([700, 800, 900])

        ax.set_xlim([dn_borders[0], dn_borders[dn_borders < last_time][-1]])
        ax.set_ylim([650, 970])

        ax.set_ylabel('EOD frequency [Hz]', fontsize = 12)
        # ax.set_xlabel('Datum')

        time_ticks = np.arange(110 * 60 + 18 * 60 * 60, last_time, 24 * 60 * 60)
        ax.set_xticklabels([])
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(['day 1', 'day 2', 'day 3', 'day 4', 'day 5', 'day 6', 'day 7', 'day 8', 'day 9', 'day 10'])
        ax.tick_params(labelsize=10)

        fig.savefig('/home/raab/Desktop/traces_big.jpg', dpi=300)
        plt.show()

    return fish_freqs

def extract_bin_freqs(datafile, shift, fish_nr_in_rec, bin_start = 0, bw = 300):
    print('extracting bin frequencies')
    current_bin_freq = [[] for f in fish_nr_in_rec]
    bin_freq = [[] for f in fish_nr_in_rec]
    centers = []

    for datei_nr in range(len(datafile)):
        fund_v, ident_v, idx_v, times_v, sign_v = loaddata(datafile[datei_nr])
        times_v += shift[datei_nr]

        ###
        while True:
            for fish_nr in range(len(fish_nr_in_rec)):
                current_bin_freq[fish_nr].extend(fund_v[(ident_v == fish_nr_in_rec[fish_nr][datei_nr]) &
                                                        (times_v >= bin_start) &
                                                        (times_v < bin_start + bw)])
            if bin_start + bw > times_v[-1]:
                break
            else:
                for fish_nr in range(len(fish_nr_in_rec)):
                    bin_freq[fish_nr].append(current_bin_freq[fish_nr])
                    current_bin_freq[fish_nr] = []
                    # print([len(doi[i]) for i in range(len(doi))])
                centers.append(bin_start + bw / 2)
                bin_start += bw

    return bin_freq, centers

def extract_bin_sign(datafile, shift, fish_nr_in_rec, bin_start = 1200, bw = 1800):
    print('extracting bin signatures')
    current_bin_sign = [[] for f in fish_nr_in_rec    ]
    bin_sign = [[] for f in fish_nr_in_rec]
    centers = []

    for datei_nr in range(len(datafile)):
        fund_v, ident_v, idx_v, times_v, sign_v = loaddata(datafile[datei_nr])
        times_v += shift[datei_nr]

        while True:
            for fish_nr in range(len(fish_nr_in_rec)):
                current_bin_sign[fish_nr].extend(sign_v[(ident_v == fish_nr_in_rec[fish_nr][datei_nr]) &
                                                        (times_v >= bin_start) &
                                                        (times_v < bin_start + bw)])
            if bin_start + bw > times_v[-1]:
                break
            else:
                for fish_nr in range(len(fish_nr_in_rec)):
                    for line in np.arange(len(current_bin_sign[fish_nr])-1)+1:
                        if np.isnan(current_bin_sign[fish_nr][line][0]):
                            current_bin_sign[fish_nr][line] = current_bin_sign[fish_nr][line]

                    bin_sign[fish_nr].append(current_bin_sign[fish_nr])
                    current_bin_sign[fish_nr] = []
                    # print([len(doi[i]) for i in range(len(doi))])
                centers.append(bin_start + bw / 2)
                bin_start += bw

    return bin_sign, centers

def extract_freq_and_pos_array(datafile, shift, fish_nr_in_rec, datafile_nr):
    fund_v, ident_v, idx_v, times_v, sign_v = loaddata(datafile)
    times_v += shift[datafile_nr]

    i_range = np.arange(0, np.nanmax(idx_v) + 1)
    # i_range = np.arange(len(np.unique(times_v)))

    fish_freqs = [np.full(len(i_range), np.nan) for i in range(len(fish_nr_in_rec))]
    fish_pos = [np.full(len(i_range), np.nan) for i in range(len(fish_nr_in_rec))]

    for fish_nr in range(len(np.array(fish_nr_in_rec)[:, datafile_nr])):
        if np.isnan(fish_nr_in_rec[fish_nr][datafile_nr]):
            continue

        freq = fund_v[ident_v == fish_nr_in_rec[fish_nr][datafile_nr]]
        pos = np.argmax(sign_v[ident_v == fish_nr_in_rec[fish_nr][datafile_nr]], axis=1)

        idx = idx_v[ident_v == fish_nr_in_rec[fish_nr][datafile_nr]]


        filled_f = np.interp(np.arange(idx[0], idx[-1] + 1), idx, freq)
        filled_p = np.interp(np.arange(idx[0], idx[-1] + 1), idx, pos)
        filled_p = np.round(filled_p, 0)

        fish_freqs[fish_nr][idx[0]:idx[-1] + 1] = filled_f
        fish_pos[fish_nr][idx[0]:idx[-1] + 1] = filled_p

    fish_freqs = np.array(fish_freqs)
    fish_pos = np.array(fish_pos)

    if len(fish_freqs[0]) != len(np.unique(times_v)):
        # ToDo: look into this ....
        # print('length of arrays dont match in %s' % datafile)
        # print('adjusting...')
        fish_freqs = fish_freqs[:, :len(np.unique(times_v))]
        fish_pos = fish_pos[:, :len(np.unique(times_v))]
        # fish_freqs = fish_freqs[:]

    return  fish_freqs, fish_pos, np.unique(times_v)

def cohans_d(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    d = (mean_x - mean_y) / np.sqrt( (np.sum((x-mean_x)**2) + np.sum((y-mean_y)**2)) / (len(x) + len(y) -2))
    return d

def main():
    ################################################################################
    ### define datafiles, shifts for each datafile, fish numbers in recordings
    if os.path.exists('/home/raab'):
        path_start = '/home/raab'
    elif os.path.exists('/home/wurm'):
        path_start = '/home/wurm'
    elif os.path.exists('/home/linhart'):
        path_start = '/home/linhart'
    else:
        path_start = ''
        print('no data found ... new user ? contact Till Raab / check connection to server')
        quit()

    saving_folder = path_start + '/analysis/'
    # datafile=[path_start + '/data/kraken_link/2018-05-04-13_10',
    #           path_start + '/data/kraken_link/2018-05-04-14:46',
    #           path_start + '/data/kraken_link/2018-05-04-16:44',
    #           path_start + '/data/kraken_link/2018-05-05-09:17',
    #           path_start + '/data/kraken_link/2018-05-06-12:14',
    #           path_start + '/data/kraken_link/2018-05-06-22:04',
    #           path_start + '/data/kraken_link/2018-05-07-08:55',
    #           path_start + '/data/kraken_link/2018-05-07-12:06',
    #           path_start + '/data/kraken_link/2018-05-07-16:14',
    #           path_start + '/data/kraken_link/2018-05-08-10:20',
    #           path_start + '/data/kraken_link/2018-05-08-17:50',
    #           path_start + '/data/kraken_link/2018-05-09-09:24',
    #           path_start + '/data/kraken_link/2018-05-09-16:53',
    #           path_start + '/data/kraken_link/2018-05-10-09:30',
    #           path_start + '/data/kraken_link/2018-05-10-20:04',
    #           path_start + '/data/kraken_link/2018-05-11-09:49',
    #           path_start + '/data/kraken_link/2018-05-11-20:21',
    #           path_start + '/data/kraken_link/2018-05-12-09:43',
    #           path_start + '/data/kraken_link/2018-05-12-19:25',
    #           path_start + '/data/kraken_link/2018-05-13-09:49',
    #           path_start + '/data/kraken_link/2018-05-13-20:08',
    #           path_start + '/data/kraken_link/2018-05-14-10:04',
    #           path_start + '/data/kraken_link/2018-05-22-14:21',
    #           path_start + '/data/kraken_link/2018-05-29-13:35']

    datafile=[path_start + '/data/2018_habitat_preference/2018-05-04-13_10',
              path_start + '/data/2018_habitat_preference/2018-05-04-14:46',
              path_start + '/data/2018_habitat_preference/2018-05-04-16:44',
              path_start + '/data/2018_habitat_preference/2018-05-05-09:17',
              path_start + '/data/2018_habitat_preference/2018-05-06-12:14',
              path_start + '/data/2018_habitat_preference/2018-05-06-22:04',
              path_start + '/data/2018_habitat_preference/2018-05-07-08:55',
              path_start + '/data/2018_habitat_preference/2018-05-07-12:06',
              path_start + '/data/2018_habitat_preference/2018-05-07-16:14',
              path_start + '/data/2018_habitat_preference/2018-05-08-10:20',
              path_start + '/data/2018_habitat_preference/2018-05-08-17:50',
              path_start + '/data/2018_habitat_preference/2018-05-09-09:24',
              path_start + '/data/2018_habitat_preference/2018-05-09-16:53',
              path_start + '/data/2018_habitat_preference/2018-05-10-09:30',
              path_start + '/data/2018_habitat_preference/2018-05-10-20:04',
              path_start + '/data/2018_habitat_preference/2018-05-11-09:49',
              path_start + '/data/2018_habitat_preference/2018-05-11-20:21',
              path_start + '/data/2018_habitat_preference/2018-05-12-09:43',
              path_start + '/data/2018_habitat_preference/2018-05-12-19:25',
              path_start + '/data/2018_habitat_preference/2018-05-13-09:49',
              path_start + '/data/2018_habitat_preference/2018-05-13-20:08',
              path_start + '/data/2018_habitat_preference/2018-05-14-10:04']

    # shift = [0, 5760, 12840, 72420, 169440, 204840, 243900, 255360, 270240, 335400, 362400, 418440, 445380, 505200, 543240, 592740, 630660, 678780, 713700, 765540, 802680, 852840, 1559460, 2161500]
    shift = [0, 5760, 12840, 72420, 169440, 204840, 243900, 255360, 270240, 335400, 362400, 418440, 445380, 505200, 543240, 592740, 630660, 678780, 713700, 765540, 802680, 852840]
    # temp = [26, 26, 26, 26.3, 26.0, 26.0, 25.8, 25.8, 25.8, 25.4, 25.4, 25.2, 25.2, 25.1, 25.1, 25.0, 25.0, 24.9, 24.9, 24.8, 24.8, 24.9, 26.3, 27.0]
    temp = [26, 26, 26, 26.3, 26.0, 26.0, 25.8, 25.8, 25.8, 25.4, 25.4, 25.2, 25.2, 25.1, 25.1, 25.0, 25.0, 24.9, 24.9, 24.8, 24.8, 24.9]
    fish_nr_in_rec = [
        [7314, 2071, 107834, 157928, 8, 2, 18372, 4, 4, 0, 7, 5, 6, 50283, 21, 28, 7, 11, 19, 76, 0, 0],
        [88, 3541, 107833, 158010, 16501, 8, 17287, 26, 32478, 1, 31, 2, 11, 4, 29496, 6, 19, 37560, 24, 3, 37192, 4],
        [7315, 9103, 107256, 158179, 3, 45, 7, 3, 3, 25208, 32881, 38054, 47218, 66437, 9402, 56948, 6, 50447, 90962, 45002, 217, 3],
        [4627, 9102, 107832, 158205, 1, 3, 2514, 2, 10, 32, 47, 25482, 12638, 66841, 53, 56949, 25745, 57594, 24839, 62328, 6, 24409],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 38916, 8, 46503, 15, 26, 9, 57152, 75735, 45, 24367, 7],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 23554, 38328, np.nan, 2, 4, 41729, 55107, 7, 84, 16706, 3810],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3155, 2144, 12, 2, 7, 117, 1],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1104, 18, 5, 10973, 57578, 42, 81580, 86637, 21],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4516, 8, 4, 3, 1, 25, 11411, 3, 57579, 21618, 247, 28786, 2],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 65093, 59600, 44, 0, 42932, 6, 108, 8, 39100],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 15004, 24342, 27327, 34423, 2, 1099, 4, 31613, 8, 7865, 4272, 57593, 3394, 74472, 3, 12],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 39778, np.nan, 1227, 2, 6, 59560, 1878, 81, 57592, np.nan, 29543, 16994, 37650],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 56947, 38877, 8, 34, 12405, 388, 25536],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 17544, 47, 31, 14, 26840, 10, 63, 48125, 146, 56950, 39918, 6, 25858, 6, 88393, 189]]

    # Q10(datafile, fish_nr_in_rec)


    colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', '#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF']

    # create_plot(datafile, shift, fish_nr_in_rec, colors)

    start_n = 110 * 60
    end_n = 110 * 60 + 12 * 60 * 60
    day_sec = 24 * 60 * 60

    habitats = [[12, 8], [14, 10], [11, 15], [9, 13], [0, 1, 2, 3, 4, 5, 6, 7]]
    hab_colors = ['k', 'grey', 'green', 'yellow', 'lightblue']


    for datafile_nr in range(len(datafile)):
        fish_f, fish_p, t = extract_freq_and_pos_array(datafile[datafile_nr], shift, fish_nr_in_rec, datafile_nr)
        if datafile_nr == 0:
            fish_freqs = fish_f
            fish_pos = fish_p
            times = t
            # print(len(t), len(fish_f[0]))
        else:
            fish_freqs = np.append(fish_freqs, fish_f, axis=1)
            fish_pos = np.append(fish_pos, fish_p, axis=1)
            times = np.append(times, t)

    clock_sec = (times % day_sec)

    night_mask = np.arange(len(clock_sec))[(clock_sec >= start_n) & (clock_sec < end_n)]
    day_mask = np.arange(len(clock_sec))[(clock_sec < start_n) | (clock_sec >= end_n)]

    dn_borders = np.arange(110 * 60, 2250000 + 12 * 60 * 60, 12 * 60 * 60)


    # how often fish on electrode x ...
    fish_counts_on_electrode = np.zeros((len(fish_nr_in_rec), 16), dtype = int)
    n_fish_counts_on_electrode = np.zeros((len(fish_nr_in_rec), 16), dtype = int)
    d_fish_counts_on_electrode = np.zeros((len(fish_nr_in_rec), 16), dtype = int)

    sep_fish_counts_on_electrode = np.array([fish_counts_on_electrode for i in dn_borders[:-1]])

    for fish_nr in range(len(fish_nr_in_rec)):
        for e_nr in range(16):
            for i in range(len(dn_borders)-1):
                sep_fish_counts_on_electrode[i][fish_nr][e_nr] += len(fish_pos[fish_nr][(fish_pos[fish_nr] == e_nr) & (times >= dn_borders[i]) & (times < dn_borders[i+1])])
            fish_counts_on_electrode[fish_nr][e_nr] += len(fish_pos[fish_nr][fish_pos[fish_nr] == e_nr])

            n_fish_counts_on_electrode[fish_nr][e_nr] += len(fish_pos[fish_nr][night_mask][fish_pos[fish_nr][night_mask] == e_nr])
            d_fish_counts_on_electrode[fish_nr][e_nr] += len(fish_pos[fish_nr][day_mask][fish_pos[fish_nr][day_mask] == e_nr])

    # how often fish in habitat x ...
    fish_counts_in_habitat = np.zeros((len(fish_nr_in_rec), 5), dtype=int)
    n_fish_counts_in_habitat = np.zeros((len(fish_nr_in_rec), 5), dtype=int)
    d_fish_counts_in_habitat = np.zeros((len(fish_nr_in_rec), 5), dtype=int)

    sep_fish_counts_in_habitat = np.array([fish_counts_in_habitat for i in dn_borders[:-1]])

    for fish_nr in range(len(fish_counts_on_electrode)):
        for habitat_nr in range(len(habitats)):
            count_in_habitat = 0
            count_in_habitat_n = 0
            count_in_habitat_d = 0
            for ele in habitats[habitat_nr]:
                count_in_habitat += fish_counts_on_electrode[fish_nr][ele]
                count_in_habitat_n += n_fish_counts_on_electrode[fish_nr][ele]
                count_in_habitat_d += d_fish_counts_on_electrode[fish_nr][ele]

            fish_counts_in_habitat[fish_nr][habitat_nr] = count_in_habitat
            d_fish_counts_in_habitat[fish_nr][habitat_nr] = count_in_habitat_d
            n_fish_counts_in_habitat[fish_nr][habitat_nr] = count_in_habitat_n

    for dn_nr in range(len(sep_fish_counts_on_electrode)):
        for fish_nr in range(len(sep_fish_counts_on_electrode[dn_nr])):
            for habitat_nr in range(len(habitats)):
                count_in_habitat = 0
                for ele in habitats[habitat_nr]:
                    count_in_habitat += sep_fish_counts_on_electrode[dn_nr][fish_nr][ele]
                sep_fish_counts_in_habitat[dn_nr][fish_nr][habitat_nr] = count_in_habitat

    fish_counts_in_habitat = np.array(fish_counts_in_habitat)
    d_fish_counts_in_habitat = np.array(d_fish_counts_in_habitat)
    n_fish_counts_in_habitat = np.array(n_fish_counts_in_habitat)

    sep_fish_counts_in_habitat = np.array(sep_fish_counts_in_habitat)
    rel_sep_fish_counts_in_habitat = np.zeros(np.shape(sep_fish_counts_in_habitat))

    for dn_nr in range(len(sep_fish_counts_in_habitat)):
        for fish_nr in range(len(sep_fish_counts_in_habitat[dn_nr])):
            if np.sum(sep_fish_counts_in_habitat[dn_nr][fish_nr]) == 0:
                pass
            else:
                # embed()
                # quit()
                rel_sep_fish_counts_in_habitat[dn_nr][fish_nr] = sep_fish_counts_in_habitat[dn_nr][fish_nr] / np.sum(sep_fish_counts_in_habitat[dn_nr][fish_nr])

    # plot for sep fish progression in habitat occupation

    # for fish_nr in range(np.shape(rel_sep_fish_counts_in_habitat)[1]):
    #     fig, ax = plt.subplots(1, 2, facecolor='white', figsize= (20/2.54, 12/2.54))
    #     for enu, dn_nr in enumerate(np.arange(0, len(rel_sep_fish_counts_in_habitat), 2)):
    #         upshift = 0
    #         for hab_nr in range(len(rel_sep_fish_counts_in_habitat[dn_nr][fish_nr])):
    #             ax[0].bar(enu, rel_sep_fish_counts_in_habitat[dn_nr][fish_nr][hab_nr], bottom=upshift, color=hab_colors[hab_nr])
    #             upshift+= rel_sep_fish_counts_in_habitat[dn_nr][fish_nr][hab_nr]
    #     ax[0].set_xlabel('night nr.')
    #     ax[0].set_ylabel('rel. occupation')
    #     ax[0].set_title('fish Nr. %.0f' % fish_nr)
    #     # ax[0].set_ylim([0, 1])
    #
    #     for enu, dn_nr in enumerate(np.arange(1, len(rel_sep_fish_counts_in_habitat), 2)):
    #         upshift = 0
    #         for hab_nr in range(len(rel_sep_fish_counts_in_habitat[dn_nr][fish_nr])):
    #             ax[1].bar(enu, rel_sep_fish_counts_in_habitat[dn_nr][fish_nr][hab_nr], bottom=upshift, color=hab_colors[hab_nr])
    #             upshift+= rel_sep_fish_counts_in_habitat[dn_nr][fish_nr][hab_nr]
    #     ax[1].set_xlabel('day nr.')
    #     ax[1].set_ylabel('rel. ocupation')
    #     plt.tight_layout()


    rel_fish_counts_in_habitat = np.zeros(np.shape(fish_counts_in_habitat))
    rel_d_fish_counts_in_habitat = np.zeros(np.shape(d_fish_counts_in_habitat))
    rel_n_fish_counts_in_habitat = np.zeros(np.shape(n_fish_counts_in_habitat))

    for fish_nr in range(len(fish_counts_in_habitat)):
        if np.sum(fish_counts_in_habitat[fish_nr]) == 0:
            pass
        else:
            rel_fish_counts_in_habitat[fish_nr] = fish_counts_in_habitat[fish_nr] / np.sum(fish_counts_in_habitat[fish_nr])

        if np.sum(d_fish_counts_in_habitat[fish_nr]) == 0:
            pass
        else:
            rel_d_fish_counts_in_habitat[fish_nr] = d_fish_counts_in_habitat[fish_nr] / np.sum(d_fish_counts_in_habitat[fish_nr])

        if np.sum(n_fish_counts_in_habitat[fish_nr]) == 0:
            pass
        else:
            rel_n_fish_counts_in_habitat[fish_nr] = n_fish_counts_in_habitat[fish_nr] / np.sum(n_fish_counts_in_habitat[fish_nr])

    embed()
    quit()
    ################################## paper figures #################################
    # transition times
    fish_hab = []
    for fish_nr in range(len(fish_pos)):
        fish_hab.append(np.full(len(fish_pos[fish_nr]), np.nan))
        for enu, hab in enumerate(habitats):
            for Chab in hab:
                fish_hab[fish_nr][fish_pos[fish_nr] == Chab] = enu

    fish_hab = np.array(fish_hab)

    transition_times = []
    d_transition_times = []
    night_d_transition_times = []
    day_d_transition_times = []

    for fish_nr in range(len(fish_hab)):
        transition_times.append(times[1:][np.abs(np.diff(fish_hab[fish_nr])) > 0])
        d_transition_times.append(np.diff(transition_times[fish_nr]))

        Ctimes_to_n_start = (np.array(transition_times[fish_nr])[:-1] - 20 * 60) % (24 * 60 * 60)
        day_mask = np.arange(len(Ctimes_to_n_start), dtype=int)[Ctimes_to_n_start > 12 * 60 * 60]
        night_mask = np.arange(len(Ctimes_to_n_start), dtype=int)[Ctimes_to_n_start < 12 * 60 * 60]
        day_d_transition_times.append(d_transition_times[fish_nr][day_mask])
        night_d_transition_times.append(d_transition_times[fish_nr][night_mask])

    dn_hab_changes = []

    for fish_nr in range(len(transition_times)):
        dn_hab_changes.append([])
        for b, e in zip(dn_borders[:-1], dn_borders[1:]):
            dn_hab_changes[fish_nr].append(len(transition_times[fish_nr][(transition_times[fish_nr] >= b) & (transition_times[fish_nr] < e)]))

    dn_hab_changes = np.array(dn_hab_changes)

    n_hab_changes = []
    d_hab_changes = []
    mdhc = []
    mnhc = []
    sdhc = []
    snhc = []
    for fish_nr in range(len(dn_hab_changes)):
        n_hab_changes.append(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][::2] != 0) & (dn_hab_changes[fish_nr][1::2] != 0)])
        mnhc.append(np.mean(n_hab_changes[fish_nr]))
        snhc.append(np.std(n_hab_changes[fish_nr]))

        d_hab_changes.append(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][::2] != 0) & (dn_hab_changes[fish_nr][1::2] != 0)])
        mdhc.append(np.mean(d_hab_changes[fish_nr]))
        sdhc.append(np.std(d_hab_changes[fish_nr]))

    # embed()
    # quit()
    # REVIEW ADAPTATIONS # 1
    ##############################################
    # MALEFEMALE

    hab_fish_count = []
    hab_male_count = []
    hab_female_count = []

    sco = 6

    fish_in_group_size = np.full(np.shape(fish_hab), np.nan) # wir n other fish (m and f)
    male_with_n_male = np.full((sco, np.shape(fish_hab)[1]), np.nan) # with n other male
    male_with_n_female = np.full((sco, np.shape(fish_hab)[1]), np.nan) # with n other male

    female_with_n_female = np.full((14-sco, np.shape(fish_hab)[1]), np.nan) # with n other female
    female_with_n_male = np.full((14-sco, np.shape(fish_hab)[1]), np.nan) # with n other female

    for hab in range(5):
        hab_fish_count.append(np.sum(np.array(fish_hab) == hab, axis= 0))
        help = np.sum(np.array(fish_hab) == hab, axis= 0)[np.where(fish_hab == hab)[1]]
        fish_in_group_size[np.where(fish_hab == hab)] = help

        hab_male_count.append(np.sum(np.array(fish_hab[:sco]) == hab, axis= 0))
        help = np.sum(np.array(fish_hab)[:sco] == hab, axis= 0)[np.where(fish_hab[:sco] == hab)[1]]
        male_with_n_male[np.where(fish_hab[:sco] == hab)] = help

        help = np.sum(np.array(fish_hab)[:sco] == hab, axis= 0)[np.where(fish_hab[sco:] == hab)[1]]
        female_with_n_male[np.where(fish_hab[sco:] == hab)] = help

        hab_female_count.append(np.sum(np.array(fish_hab[sco:]) == hab, axis= 0))
        help = np.sum(np.array(fish_hab)[sco:] == hab, axis= 0)[np.where(fish_hab[sco:] == hab)[1]]
        female_with_n_female[np.where(fish_hab[sco:] == hab)] = help

        help = np.sum(np.array(fish_hab)[sco:] == hab, axis= 0)[np.where(fish_hab[:sco] == hab)[1]]
        male_with_n_female[np.where(fish_hab[:sco] == hab)] = help


    i_of_plus_fish = [0]
    i_of_plus_fish.append(np.arange(len(hab_fish_count[0]))[np.sum(hab_fish_count, axis=0) > 4][0])
    i_of_plus_fish.append(np.arange(len(hab_fish_count[0]))[np.sum(hab_fish_count, axis=0) > 6][0])
    i_of_plus_fish.append(np.arange(len(hab_fish_count[0]))[np.sum(hab_fish_count, axis=0) > 8][0])
    i_of_plus_fish.append(np.arange(len(hab_fish_count[0]))[np.sum(hab_fish_count, axis=0) > 10][0])
    i_of_plus_fish.append(np.arange(len(hab_fish_count[0]))[np.sum(hab_fish_count, axis=0) > 12][0])
    i_of_plus_fish.append(len(hab_fish_count[0]))

    count_m = [4, 4, 4, 6, 6, 6]
    count_f = [0, 2, 4, 4, 6, 8]

    n_mask = np.full(len(times), False)
    for ns, ne in zip(dn_borders[::2], dn_borders[1::2]):
        n_mask[(times >= ns) & (times < ne)] = True



    hab_change = [] # [fish ID],[time, origin, destination]
    for fish_id in range(len(fish_hab)):
        mask = np.diff(fish_hab[fish_id][~np.isnan(fish_hab[fish_id])] != 0)
        change_time = times[~np.isnan(fish_hab[fish_id])][:-1][mask]
        origin = fish_hab[fish_id][~np.isnan(fish_hab[fish_id])][:-1][mask]
        destination = fish_hab[fish_id][~np.isnan(fish_hab[fish_id])][1:][mask]
        hab_change.append([change_time, origin, destination])

    # ToDo: additional list for individuals !!!
    fish_dts_night = []
    fish_dts_day = []
    for fish_id in tqdm(np.arange(len(fish_hab))):
        dts_day = []
        dts_night = []
        for i in tqdm(np.arange(len(hab_change[fish_id][0]))):
            for c_fish_id in np.arange(len(fish_hab)):
                if fish_id == c_fish_id:
                    continue

                if hab_change[fish_id][2][i] == 4:
                    continue

                dt = hab_change[c_fish_id][0][ (hab_change[c_fish_id][0] > hab_change[fish_id][0][i]) &
                                               (hab_change[c_fish_id][0] <= hab_change[fish_id][0][i] + 10) &
                                               (hab_change[c_fish_id][1] == hab_change[fish_id][2][i]) &
                                               (hab_change[c_fish_id][2] != 4)]
                if len(dt >= 1):

                    dt = dt[0] - hab_change[fish_id][0][i]
                    if n_mask[i] == True:
                        dts_night.append(dt)
                    elif n_mask[i] == False:
                        dts_day.append(dt)
        fish_dts_day.append(dts_day)
        fish_dts_night.append(dts_night)

    boot_night = []
    boot_day = []

    night_indices = np.arange(len(n_mask))[n_mask]
    day_indices = np.arange(len(n_mask))[~n_mask]

    # ToDo: rethink this...
    stacked_change_times = np.hstack(np.array(hab_change)[:, 0])
    stacked_change_hab = np.hstack(np.array(hab_change)[:, 1])

    t0= time.time()
    for enu in range(2):
        if enu == 0:
            while len(boot_night) < 20000:
                i = np.random.randint(0, len(night_indices), 1)[0]
                chab = np.random.randint(0, 4, 1)[0]
                ctime = stacked_change_times[(stacked_change_times > times[i]) &
                                             (stacked_change_times < times[i] + 10) &
                                             (stacked_change_hab == chab)]
                if len(ctime) > 0:
                    dt = ctime[0] - times[i]
                    boot_night.append(dt)
        else:
            while len(boot_day) < 20000:
                i = np.random.randint(0, len(day_indices), 1)[0]
                chab = np.random.randint(0, 4, 1)[0]
                ctime = stacked_change_times[(stacked_change_times > times[i]) &
                                             (stacked_change_times < times[i] + 10) &
                                             (stacked_change_hab == chab)]
                if len(ctime) > 0:
                    dt = ctime[0] - times[i]
                    boot_day.append(dt)




    fig, ax = plt.subplots(1, 2, figsize=(20/2.54, 12/2.54), facecolor='white', sharey=True)
    ax = np.hstack(ax)
    n_day =  []
    n_night = []
    for i in range(6):
        n, bins = np.histogram(fish_dts_day[i], bins=np.arange(times[0] + (times[1]-times[0]) / 2, 10, times[1]-times[0]))
        n = n / np.sum(n) / (times[1] - times[0])
        n_day.append(n)
        ax[0].plot(bins[:-1] + (times[1]-times[0]) / 2, n, color='cornflowerblue', lw = 1)

        n, bins = np.histogram(fish_dts_night[i], bins=np.arange(times[0] + (times[1]-times[0]) / 2, 10, times[1]-times[0]))
        n = n / np.sum(n) / (times[1] - times[0])
        n_night.append(n)
        ax[0].plot(bins[:-1] + (times[1]-times[0]) / 2, n, color='blue', lw = 1)


    for i in np.arange(6, 14):

        n, bins = np.histogram(fish_dts_day[i], bins=np.arange(times[0] + (times[1]-times[0]) / 2, 10, times[1]-times[0]))
        n = n / np.sum(n) / (times[1] - times[0])
        n_day.append(n)
        ax[1].plot(bins[:-1] + (times[1]-times[0]) / 2, n, color='pink', lw = 1)

        n, bins = np.histogram(fish_dts_night[i], bins=np.arange(times[0] + (times[1]-times[0]) / 2, 10, times[1]-times[0]))
        n = n / np.sum(n) / (times[1] - times[0])
        n_night.append(n)
        ax[1].plot(bins[:-1] + (times[1]-times[0]) / 2, n, color='firebrick', lw = 1)

    ax[0].set_xlabel('$\Delta$t [s]', fontsize=10)
    ax[1].set_xlabel('$\Delta$t [s]', fontsize=10)
    ax[0].set_ylabel('Probability', fontsize=10)

    n, bins = np.histogram(boot_day, bins=np.arange(times[0] + (times[1] - times[0]) / 2, 10, times[1] - times[0]))
    n = n / np.sum(n) / (times[1] - times[0])
    boot_n_day = n
    ax[0].plot(bins[:-1] + (times[1]-times[0]) / 2, n, '--', color='grey')
    ax[1].plot(bins[:-1] + (times[1]-times[0]) / 2, n, '--', color='grey')

    n, bins = np.histogram(boot_night, bins=np.arange(times[0] + (times[1] - times[0]) / 2, 10, times[1] - times[0]))
    n = n / np.sum(n) / (times[1] - times[0])
    boot_n_night = n
    ax[0].plot(bins[:-1] + (times[1]-times[0]) / 2, n, '--', color='k')
    ax[1].plot(bins[:-1] + (times[1]-times[0]) / 2, n, '--', color='k')

    if True:
        print('\n KS-Test dt to leaving fish after enterence')
        # print('day')
        for i in range(len(fish_dts_day)):
            if len(fish_dts_day[i]) > 0:
                D, p = scp.ks_2samp(fish_dts_day[i], boot_day)
                d = cohans_d(fish_dts_day[i], boot_day)
                print('\nday   / fish%.0f: D = %.2f, p = %.3f, d = %.2f' % (i+1, D, p, d))
            else:
                print('\nday   / fish%.0f: n.a.' % (i+1))

            if len(fish_dts_night[i]) > 0:
                D, p = scp.ks_2samp(fish_dts_night[i], boot_day)
                d = cohans_d(fish_dts_night[i], boot_day)
                print('night / fish%.0f: D = %.2f, p = %.3f, d = %.2f' % (i+1, D, p, d))
            else:
                print('night / fish%.0f: n.a.' % (i+1))


    ###############################################


    d_male_grouping = [[], [], [], [], [], []]
    n_male_grouping = [[], [], [], [], [], []]

    d_female_grouping = [[], [], [], [], [], []]
    n_female_grouping = [[], [], [], [], [], []]
    for n in np.arange(len(fish_hab))+1:

        d_male_grouping[0].append(len(np.hstack(np.array(hab_male_count)[:, ~n_mask])[np.hstack(np.array(hab_male_count)[:, ~n_mask]) == n]))
        n_male_grouping[0].append(len(np.hstack(np.array(hab_male_count)[:, n_mask])[np.hstack(np.array(hab_male_count)[:, n_mask]) == n]))

        d_female_grouping[0].append(len(np.hstack(np.array(hab_female_count)[:, ~n_mask])[np.hstack(np.array(hab_female_count)[:, ~n_mask]) == n]))
        n_female_grouping[0].append(len(np.hstack(np.array(hab_female_count)[:, n_mask])[np.hstack(np.array(hab_female_count)[:, n_mask]) == n]))

        for hab_nr in range(5):
            d_male_grouping[hab_nr+1].append(len(hab_male_count[hab_nr][~n_mask][hab_male_count[hab_nr][~n_mask] == n]))
            n_male_grouping[hab_nr+1].append(len(hab_male_count[hab_nr][n_mask][hab_male_count[hab_nr][n_mask] == n]))


            d_female_grouping[hab_nr+1].append(len(hab_female_count[hab_nr][~n_mask][hab_female_count[hab_nr][~n_mask] == n]))
            n_female_grouping[hab_nr+1].append(len(hab_female_count[hab_nr][n_mask][hab_female_count[hab_nr][n_mask] == n]))

    d_male_grouping = np.array(d_male_grouping)
    n_male_grouping = np.array(n_male_grouping)

    d_female_grouping = np.array(d_female_grouping)
    n_female_grouping = np.array(n_female_grouping)


    #######################
    n = np.arange(len(fish_hab))+1
    male_in_groupsize = []
    female_in_groupsize = []

    for hab_nr in range(5):
        male_in_groupsize.append([])
        female_in_groupsize.append([])
        for t0, t1 in zip(i_of_plus_fish[:-1], i_of_plus_fish[1:]):
            male_in_groupsize[-1].append([])
            female_in_groupsize[-1].append([])

            mc = np.array(hab_male_count)[hab_nr, t0:t1]
            fc = np.array(hab_female_count)[hab_nr, t0:t1]
            m = []
            f = []
            for i in n:
                for j in range(i):
                    m.extend(np.array(mc+fc)[mc == i])
                    f.extend(np.array(mc+fc)[fc == i])
            male_in_groupsize[-1][-1].extend(m)
            female_in_groupsize[-1][-1].extend(f)

    fig, ax = plt.subplots(5, 1, figsize=(20/2.54, 20/2.54), facecolor='white')
    ax = np.hstack(ax)
    ax2 = []
    for i in range(len(male_in_groupsize)):
        axx = ax[i].twinx()
        ax2.append(axx)

        ax[i].plot(np.arange(len(male_in_groupsize[i])), (np.array(count_m) + np.array(count_f)) / 5, '--', lw =1, color='k')

        ax[i].errorbar(np.arange(len(male_in_groupsize[i])) - 0.1, list(map(lambda x: np.mean(x), male_in_groupsize[i])), yerr=list(map(lambda x: np.std(x), male_in_groupsize[i])), fmt='none', ecolor='blue')
        ax[i].plot(np.arange(len(male_in_groupsize[i])) - 0.1, list(map(lambda x: np.mean(x), male_in_groupsize[i])), 'o', color='blue')

        help = np.copy(male_in_groupsize[i])
        for j, fish_c in zip(range(len(help)), np.array(count_m) + np.array(count_f)):
            help[j] = np.array(help[j]) / fish_c
        ax2[i].errorbar(6 - 0.1, np.mean(np.hstack(help)), yerr=np.std(np.hstack(help)), fmt='none', ecolor='blue')
        ax2[i].plot(6 - 0.1, np.mean(np.hstack(help)), 'o', color='blue')

        ax[i].errorbar(np.arange(len(male_in_groupsize[i])) + 0.1, list(map(lambda x: np.mean(x), female_in_groupsize[i])), yerr=list(map(lambda x: np.std(x), female_in_groupsize[i])), fmt='none', ecolor='pink')
        ax[i].plot(np.arange(len(male_in_groupsize[i])) + 0.1, list(map(lambda x: np.mean(x), female_in_groupsize[i])), 'o', color='pink')

        help = np.copy(female_in_groupsize[i])
        for j, fish_c in zip(range(len(help)), np.array(count_m) + np.array(count_f)):
            help[j] = np.array(help[j]) / fish_c
        ax2[i].errorbar(6 + 0.1, np.mean(np.hstack(help)), yerr=np.std(np.hstack(help)), fmt='none', ecolor='pink')
        ax2[i].plot(6 + 0.1, np.mean(np.hstack(help)), 'o', color='pink')
        #
        ax2[i].set_ylim([0, 1])

        ax[i].set_ylabel('group size', fontsize=10)
        ax2[i].set_ylabel('rel.\ngroup size', fontsize=10)


        ax[i].set_xticks(np.append(np.arange(len(male_in_groupsize[0])), [5.9, 6.1]))
        ax[i].set_xticklabels([])
        ax[i].set_ylim([0, 8])
        ax[i].plot([5.5, 5.5], [0, 8], '-', lw =2, color='k')


    ax[-1].set_xticks(np.append(np.arange(len(male_in_groupsize[0])), [5.9, 6.1]))
    ax[-1].set_xlabel('fish count', fontsize=10)

    # ax[-1].set_xticks(np.arange(len(male_in_groupsize[0])))
    ax[-1].set_xticklabels(np.append(np.array(count_m) + np.array(count_f), [u'\u2642', u'\u2640']))

    plt.tight_layout()



    #######################

    ### grouping all ###
    n = np.arange(len(fish_hab)) + 1
    fig, ax = plt.subplots(3, 2, figsize=(20/2.54, 20/2.54), facecolor='white') # ToDo: for single habitats
    ax = np.hstack(ax)
    txt = ['all', 'stacked stones', 'canyon', 'plants', 'gravel', 'open water']

    for i in range(len(ax)):
        ax[i].bar(n-.15, d_male_grouping[i] / (np.sum(d_male_grouping[i]) + np.sum(d_female_grouping)), width = .1, color='cornflowerblue')
        ax[i].plot(n, d_male_grouping[i] / (np.sum(d_male_grouping[i]) + np.sum(d_female_grouping)) * n, color='cornflowerblue', marker='.', label='day male')

        ax[i].bar(n-.05, n_male_grouping[i] / (np.sum(n_male_grouping[i]) + np.sum(n_female_grouping)), width = .1, color='blue')
        ax[i].plot(n, n_male_grouping[i] / (np.sum(n_male_grouping[i]) + np.sum(n_female_grouping)) * n, color='blue', marker='.', label='night male')

        ax[i].bar(n+.05, d_female_grouping[i] / (np.sum(d_female_grouping[i]) + np.sum(d_male_grouping)), width = .1, color='pink')
        ax[i].plot(n, d_female_grouping[i] / (np.sum(d_female_grouping[i]) + np.sum(d_male_grouping)) * n, color='pink', marker='.', label='day female')

        ax[i].bar(n+.15, n_female_grouping[i] / (np.sum(n_female_grouping[i]) + np.sum(n_male_grouping)), width = .1, color='firebrick')
        ax[i].plot(n, n_female_grouping[i] / (np.sum(n_female_grouping[i]) + np.sum(n_male_grouping)) * n, color='firebrick', marker='.', label='night female')

        ax[i].text(8, .02, txt[i], va='center', ha='center')
        ax[i].set_xlim([0, 10])
        # ax[i].set_ylim([0, .3])
        ax[i].set_xticks(np.arange(10)+1)
        ax[i].legend(loc=1, frameon=False, fontsize=8)

    ax[4].set_xlabel('group size')
    ax[5].set_xlabel('group size')
    ax[0].set_ylabel('probability')
    ax[2].set_ylabel('probability')
    ax[4].set_ylabel('probability')



    group_size_per_id_night = []
    group_size_per_id_day = []
    for i in range(len(fish_in_group_size)):
        group_size_per_id_day.append(fish_in_group_size[i][~n_mask][~np.isnan(fish_in_group_size[i][~n_mask])])
        group_size_per_id_night.append(fish_in_group_size[i][n_mask][~np.isnan(fish_in_group_size[i][n_mask])])

    n_male_per_male_night = []
    n_male_per_male_day = []
    for i in range(len(male_with_n_male)):
        n_male_per_male_day.append(male_with_n_male[i][~n_mask][~np.isnan(male_with_n_male[i][~n_mask])])
        n_male_per_male_night.append(male_with_n_male[i][n_mask][~np.isnan(male_with_n_male[i][n_mask])])

    n_female_per_female_night = []
    n_female_per_female_day = []
    for i in range(len(female_with_n_female)):
        n_female_per_female_day.append(female_with_n_female[i][~n_mask][~np.isnan(female_with_n_female[i][~n_mask])])
        n_female_per_female_night.append(female_with_n_female[i][n_mask][~np.isnan(female_with_n_female[i][n_mask])])

    n_female_per_male_night = []
    n_female_per_male_day = []
    for i in range(len(male_with_n_female)):
        n_female_per_male_day.append(male_with_n_female[i][~n_mask][~np.isnan(male_with_n_female[i][~n_mask])])
        n_female_per_male_night.append(male_with_n_female[i][n_mask][~np.isnan(male_with_n_female[i][n_mask])])

    n_male_per_female_night = []
    n_male_per_female_day = []
    for i in range(len(female_with_n_male)):
        n_male_per_female_day.append(female_with_n_male[i][~n_mask][~np.isnan(female_with_n_male[i][~n_mask])])
        n_male_per_female_night.append(female_with_n_male[i][n_mask][~np.isnan(female_with_n_male[i][n_mask])])


    fig, ax = plt.subplots(3, 1, figsize=(20/2.54, 20/2.54))
    ax = np.hstack(ax)
    ax[0].errorbar(np.arange(14)+1-0.1, list(map(lambda x: np.mean(x), group_size_per_id_day)), yerr=list(map(lambda x: np.std(x), group_size_per_id_day)), fmt='none', ecolor='grey')
    ax[0].plot(np.arange(14)+1-0.1, list(map(lambda x: np.mean(x), group_size_per_id_day)), 'o', color='grey')

    ax[0].errorbar(np.arange(14)+1+0.1, list(map(lambda x: np.mean(x), group_size_per_id_night)), yerr=list(map(lambda x: np.std(x), group_size_per_id_night)), ecolor='k', fmt='none')
    ax[0].plot(np.arange(14)+1+0.1, list(map(lambda x: np.mean(x), group_size_per_id_night)), 'o', color='k')

    ax[0].set_xticks(np.arange(14) +1)
    ax[0].set_xlabel('Fish ID')
    ax[0].set_ylabel('total individuals')
    ax[0].set_title('group size and composition')


    ax[1].errorbar(np.arange(6)+1-0.1, list(map(lambda x: np.mean(x), n_male_per_male_day)), yerr=list(map(lambda x: np.std(x), n_male_per_male_day)), fmt='none', ecolor='grey')
    ax[1].plot(np.arange(6)+1-0.1, list(map(lambda x: np.mean(x), n_male_per_male_day)), 'o', color='grey')

    ax[1].errorbar(np.arange(6)+1+0.1, list(map(lambda x: np.mean(x), n_male_per_male_night)), yerr=list(map(lambda x: np.std(x), n_male_per_male_night)), ecolor='k', fmt='none')
    ax[1].plot(np.arange(6)+1+0.1, list(map(lambda x: np.mean(x), n_male_per_male_night)), 'o', color='k')



    ax[1].errorbar(np.arange(8)+1+6-0.1, list(map(lambda x: np.mean(x), n_female_per_female_day)), yerr=list(map(lambda x: np.std(x), n_female_per_female_day)), fmt='none', ecolor='grey')
    ax[1].plot(np.arange(8)+1+6-0.1, list(map(lambda x: np.mean(x), n_female_per_female_day)), 'o', color='grey')

    ax[1].errorbar(np.arange(8)+1+6+0.1, list(map(lambda x: np.mean(x), n_female_per_female_night)), yerr=list(map(lambda x: np.std(x), n_female_per_female_night)), ecolor='k', fmt='none')
    ax[1].plot(np.arange(8)+1+6+0.1, list(map(lambda x: np.mean(x), n_female_per_female_night)), 'o', color='k')

    ax[1].set_xticks(np.arange(14) +1)
    ax[1].set_xlabel('Fish ID')
    ax[1].set_ylabel('same sex\nindividuals')



    ax[2].errorbar(np.arange(6)+1-0.1, list(map(lambda x: np.mean(x), n_female_per_male_day)), yerr=list(map(lambda x: np.std(x), n_female_per_male_day)), fmt='none', ecolor='grey')
    ax[2].plot(np.arange(6)+1-0.1, list(map(lambda x: np.mean(x), n_female_per_male_day)), 'o', color='grey')

    ax[2].errorbar(np.arange(6)+1+0.1, list(map(lambda x: np.mean(x), n_female_per_male_night)), yerr=list(map(lambda x: np.std(x), n_female_per_male_night)), ecolor='k', fmt='none')
    ax[2].plot(np.arange(6)+1+0.1, list(map(lambda x: np.mean(x), n_female_per_male_night)), 'o', color='k')


    ax[2].errorbar(np.arange(8)+1+6-0.1, list(map(lambda x: np.mean(x), n_male_per_female_day)), yerr=list(map(lambda x: np.std(x), n_male_per_female_day)), fmt='none', ecolor='grey')
    ax[2].plot(np.arange(8)+1+6-0.1, list(map(lambda x: np.mean(x), n_male_per_female_day)), 'o', color='grey')

    ax[2].errorbar(np.arange(8)+1+6+0.1, list(map(lambda x: np.mean(x), n_male_per_female_night)), yerr=list(map(lambda x: np.std(x), n_male_per_female_night)), ecolor='k', fmt='none')
    ax[2].plot(np.arange(8)+1+6+0.1, list(map(lambda x: np.mean(x), n_male_per_female_night)), 'o', color='k')

    ax[2].set_xticks(np.arange(14) +1)
    ax[2].set_xlabel('Fish ID')
    ax[2].set_ylabel('diff. sex\nindividuals')

    ax[0].set_ylim([0, 7])
    ax[1].set_ylim([0, 4.5])
    ax[2].set_ylim([0, 4.5])

    plt.tight_layout()



    ratio_day = []
    ratio_night = []
    for i0, i1 in zip(i_of_plus_fish[:-1], i_of_plus_fish[1:]):
        fish_count_mask = np.arange(i0, i1)
        ratio_day.append([])
        ratio_night.append([])
        for hab_id in range(5):
            ratio_day[-1].append(hab_male_count[hab_id][fish_count_mask][~n_mask[fish_count_mask]] / (
                    hab_male_count[hab_id][fish_count_mask][~n_mask[fish_count_mask]] + hab_female_count[hab_id][fish_count_mask][~n_mask[fish_count_mask]]))

            ratio_night[-1].append(hab_male_count[hab_id][fish_count_mask][n_mask[fish_count_mask]] / (
                    hab_male_count[hab_id][fish_count_mask][n_mask[fish_count_mask]] + hab_female_count[hab_id][fish_count_mask][n_mask[fish_count_mask]]))

            ratio_day[-1][-1] = ratio_day[-1][-1][~np.isnan(ratio_day[-1][-1])]
            ratio_night[-1][-1] = ratio_night[-1][-1][~np.isnan(ratio_night[-1][-1])]

    fig, ax = plt.subplots(2, 3, figsize=(20/2.54, 12/2.54), facecolor='white')
    ax = np.hstack(ax)

    for i in np.arange(len(ratio_day)-1)+1:
        Cratio = count_m[i] / (count_m[i] + count_f[i])
        ax[i].plot([0.5, 5.5], [Cratio, Cratio], '--', lw=1, color='k')
        ax[i].errorbar(np.arange(len(ratio_day[i])) + 1 - 0.1, np.array(list(map(lambda x: np.mean(x), ratio_day[i]))), yerr=list(map(lambda x: np.std(x), ratio_day[i])), fmt='none', ecolor='grey')
        ax[i].plot(np.arange(len(ratio_day[i])) + 1 - 0.1, np.array(list(map(lambda x: np.mean(x), ratio_day[i]))), 'o', color='grey')

        ax[i].errorbar(np.arange(len(ratio_night[i])) + 1 + 0.1, np.array(list(map(lambda x: np.mean(x), ratio_night[i]))), yerr=list(map(lambda x: np.std(x), ratio_night[i])), fmt='none', ecolor='k')
        ax[i].plot(np.arange(len(ratio_night[i])) + 1 + 0.1, np.array(list(map(lambda x: np.mean(x), ratio_night[i]))), 'o', color='k')
        ax[i].set_ylim(0, 1)
        ax[i].set_xlim(0.5, 5.5)

        ax[i].text(1, .075, u'\u2642:%.0f\n\u2640:%.0f' % (count_m[i], count_f[i]), fontsize=8, va='center', ha='center')

        ax[i].set_xticks(np.arange(5) + 1)
        ax[i].set_xticklabels([])

    glob_mean_day = []
    glob_mean_night = []
    glob_std_day = []
    glob_std_night = []
    for hab_nr in range(np.shape(ratio_day)[1]):
        collect_day = []
        collect_night = []
        for i in np.arange(len(ratio_day)-1)+1:
            collect_day.extend(ratio_day[i][hab_nr] - count_m[i] / (count_m[i] + count_f[i]))
            collect_night.extend(ratio_night[i][hab_nr] - count_m[i] / (count_m[i] + count_f[i]))
        # embed()
        # quit()
        glob_mean_day.append(np.mean(collect_day))
        glob_mean_night.append(np.mean(collect_night))
        glob_std_day.append(np.std(collect_day))
        glob_std_night.append(np.std(collect_night))

    ax[0].plot([0.5, 5.5], [0, 0], '--', lw = 1, color='k')
    ax[0].errorbar(np.arange(len(ratio_day[i])) + 1 - 0.1, glob_mean_day, yerr=glob_std_day, fmt='none', ecolor='grey')
    ax[0].plot(np.arange(len(ratio_day[i])) + 1 - 0.1, glob_mean_day, 'o', color='grey')

    ax[0].errorbar(np.arange(len(ratio_night[i])) + 1 + 0.1, glob_mean_night, yerr=glob_std_night, fmt='none', ecolor='k')
    ax[0].plot(np.arange(len(ratio_night[i])) + 1 + 0.1, glob_mean_night, 'o', color='k')
    ax[0].set_ylim(-0.5, 0.5)
    ax[0].set_xlim(0.5, 5.5)

    ax[0].text(1, -0.425, 'all', fontsize=8, va='center', ha='center')
    ax[0].set_xticks(np.arange(5) + 1)
    ax[0].set_xticklabels([])

    # ax[-2].set_xticks(np.arange(5) + 1)
    ax[0].set_ylabel('male ratio', fontsize=10)
    ax[0].set_title('above expected male ratio', fontsize=10)
    ax[3].set_ylabel('male ratio', fontsize=10)
    ax[-1].set_xticklabels(['st. stones', 'iso. stones', 'grass', 'gravel', 'water'], rotation = 70)
    ax[-2].set_xticklabels(['st. stones', 'iso. stones', 'grass', 'gravel', 'water'], rotation = 70)
    ax[-3].set_xticklabels(['st. stones', 'iso. stones', 'grass', 'gravel', 'water'], rotation = 70)
    plt.tight_layout()


    #####
    rfig, rax = plt.subplots(1, 2, figsize=(20/2.54, 12/2.54), facecolor='white')
    rax = np.hstack(rax)

    Cratio = count_m[-1] / (count_m[-1] + count_f[-1])

    rax[0].errorbar(np.arange(5) - 0.1, np.array(list(map(lambda x: np.mean(x), np.array(ratio_day)[-1, :]))),
                    yerr=list(map(lambda x: np.std(x), np.array(ratio_day)[-1, :])), fmt='none', ecolor='k')
    # rax[0].plot(np.arange(5) - 0.1, np.array(list(map(lambda x: np.mean(x), np.array(ratio_day)[-1, :]))), 'o', color='grey')
    rax[0].bar(np.arange(5) - 0.1, np.array(list(map(lambda x: np.mean(x), np.array(ratio_day)[-1, :]))), width=.2, color='cornflowerblue', label='day')

    rax[0].errorbar(np.arange(5) + 0.1, np.array(list(map(lambda x: np.mean(x), np.array(ratio_night)[-1, :]))),
                    yerr=list(map(lambda x: np.std(x), np.array(ratio_night)[-1, :])), fmt='none', ecolor='k')
    # rax[0].plot(np.arange(5) + 0.1, np.array(list(map(lambda x: np.mean(x), np.array(ratio_night)[-1, :]))), 'o', color='k')
    rax[0].bar(np.arange(5) + 0.1, np.array(list(map(lambda x: np.mean(x), np.array(ratio_night)[-1, :]))), width=.2, color='#888888', label='night')
    rax[0].set_ylim(0, 1)
    rax[0].set_xlim(-0.5, 4.5)

    rax[0].text(0, .075, u'\u2642:%.0f\n\u2640:%.0f' % (count_m[-1], count_f[-1]), fontsize=10, va='center', ha='center')

    rax[0].set_xticks(np.arange(5))
    rax[0].set_xticklabels(['st. stones', 'iso. stones', 'grass', 'gravel', 'water'], rotation = 45)
    rax[0].tick_params(labelsize=9)
    rax[0].set_ylabel('male ratio', fontsize=10)
    rax[0].plot([-0.5, 4.5], [Cratio, Cratio], '--', lw=1, color='k')
    rax[0].legend(loc=1, fontsize=9, frameon=False)

    if True:
        print('\n Mann-Whitney U male ratio day-night')
        for i, hab in zip(range(5), ['st. stones', 'iso. stones', 'plants', 'gravel', 'open water']):
            stats, p = scp.mannwhitneyu(np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i])
            d = cohans_d(np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i])
            print('%s: U = %.0f, p = %.3f, d = %.2f' % (hab, stats, p, d))
        print('\n Mann-Whitney U male ratio between habitats')
        for i, hab in zip(np.arange(5), np.array(['st. stones', 'iso. stones', 'plants', 'gravel', 'open water'])):
            mean = np.mean(np.hstack([np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i]]))
            std = np.std(np.hstack([np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i]]))
            print('%s: %.2f+-%.2f' % (hab, mean, std))

            for j, hab2 in zip(np.arange(5)[i+1:], np.array(['st. stones', 'iso. stones', 'plants', 'gravel', 'open water'])[i+1:]):
                stats , p = scp.mannwhitneyu(np.hstack([np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i]]),
                                             np.hstack([np.array(ratio_day)[-1, j], np.array(ratio_night)[-1, j]]))
                d = cohans_d(np.hstack([np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i]]),
                             np.hstack([np.array(ratio_day)[-1, j], np.array(ratio_night)[-1, j]]))

                print('%s - %s: U = %.0f, p = %.3f, d = %.2f' % (hab, hab2, stats, p, d))



    rax[1].errorbar(np.arange(5) - .1, list(map(lambda x: np.mean(x), np.array(male_in_groupsize)[:, 5])),
                    yerr=list(map(lambda x: np.std(x), np.array(male_in_groupsize)[:, 5])), fmt='none', ecolor='k')
    # rax[1].plot(np.arange(5) - .1, list(map(lambda x: np.mean(x), np.array(male_in_groupsize)[:, 5])), 'o', color='blue')
    rax[1].bar(np.arange(5) - .1, list(map(lambda x: np.mean(x), np.array(male_in_groupsize)[:, 5])), width=.2, color='firebrick', label=u'\u2642')

    rax[1].errorbar(np.arange(5) + .1, list(map(lambda x: np.mean(x), np.array(female_in_groupsize)[:, 5])),
                    yerr=list(map(lambda x: np.std(x), np.array(female_in_groupsize)[:, 5])), fmt='none', ecolor='k')
    # rax[1].plot(np.arange(5) + .1, list(map(lambda x: np.mean(x), np.array(female_in_groupsize)[:, 5])), 'o', color='pink')
    rax[1].bar(np.arange(5) + .1, list(map(lambda x: np.mean(x), np.array(female_in_groupsize)[:, 5])), width=.2, color=colors[2], label=u'\u2640')

    rax[1].set_xticks(np.arange(5))
    rax[1].set_xticklabels(['st. stones', 'iso. stones', 'grass', 'gravel', 'water'], rotation = 45)
    rax[1].tick_params(labelsize=9)
    rax[1].set_ylabel('group size', fontsize=10)
    rax[1].legend(loc=1, fontsize=9, frameon=False)
    plt.tight_layout()

    if True:
        print('\n Mann-Whitney U groupsize male-female')
        for i, hab in zip(range(5), ['st. stones', 'iso. stones', 'plants', 'gravel', 'open water']):
            stats, p = scp.mannwhitneyu(np.array(male_in_groupsize)[i, 5], np.array(female_in_groupsize)[i, 5])
            d = cohans_d(np.array(male_in_groupsize)[i, 5], np.array(female_in_groupsize)[i, 5])
            print('%s: U = %.0f, p = %.3f, d = %.2f' % (hab, stats, p, d))

        print('\n Mann-Whitney U groupsize between hab')
        for i, hab in zip(range(5), ['st. stones', 'iso. stones', 'plants', 'gravel', 'open water']):
            mean = np.mean(np.hstack([np.array(male_in_groupsize)[i, 5], np.array(female_in_groupsize)[i, 5]]))
            std = np.std(np.hstack([np.array(male_in_groupsize)[i, 5], np.array(female_in_groupsize)[i, 5]]))
            print('%s: %.2f+-%.2f' % (hab, mean, std))
            for j, hab2 in zip(np.arange(5)[i + 1:], np.array(['st. stones', 'iso. stones', 'plants', 'gravel', 'open water'])[i + 1:]):
                stats, p = scp.mannwhitneyu(np.hstack([np.array(male_in_groupsize)[i, 5], np.array(female_in_groupsize)[i, 5]]),
                                            np.hstack([np.array(male_in_groupsize)[j, 5], np.array(female_in_groupsize)[j, 5]]))

                d = cohans_d(np.hstack([np.array(male_in_groupsize)[i, 5], np.array(female_in_groupsize)[i, 5]]),
                             np.hstack([np.array(male_in_groupsize)[j, 5], np.array(female_in_groupsize)[j, 5]]))
                print('%s - %s: U = %.0f, p = %.3f, d = %.2f' % (hab, hab2, stats, p, d))

    # ax.boxplot(ratio_day, positions = np.arange(len(ratio_day)) + 1 - 0.1, widths = 0.2)
    # ax.boxplot(ratio_night, positions = np.arange(len(ratio_day)) + 1 + 0.1, widths = 0.2)

    # REVIEW ADAPTATIONS # 2

    # mask = (male_count > 0) | (female_count > 0)
    #
    # m_ratio = male_count[mask] / (male_count[mask] + female_count[mask])
    # m_unique_ratio = np.unique(m_ratio)
    # n_m_unique_ratio = []
    # for r in m_unique_ratio:
    #     n_m_unique_ratio.append(len(m_ratio[m_ratio == r]))
    #
    # f_ratio = female_count[mask] / (male_count[mask] + female_count[mask])
    # f_unique_ratio = np.unique(f_ratio)
    # n_f_unique_ratio = []
    # for r in f_unique_ratio:
    #     n_f_unique_ratio.append(len(f_ratio[f_ratio == r]))
    #
    # fig, ax = plt.subplots()
    # ax.plot(m_unique_ratio, n_m_unique_ratio / np.sum(n_m_unique_ratio), color='blue')
    # ax.plot(f_unique_ratio, n_f_unique_ratio / np.sum(n_f_unique_ratio), color='pink')

    ######


    bin_start = 110 * 60
    bw = 12 * 60 * 60

    bin_t = []
    binned_hab_fish_count = [[], [], [], [], []]
    # while bin_start < times[-1]:
    for i in range(20):
        for hab in range(5):
            Cbin_data = hab_fish_count[hab][(times >= bin_start) & (times < bin_start + bw)]
            binned_hab_fish_count[hab].append(np.mean(Cbin_data))
        bin_t.append(bin_start + bw / 2)
        bin_start += bw

    fish_pref_hab = []
    fish_2pref_hab = []
    fish_dn_mask = []
    fish_perc_in_pref_hab = []
    fish_perc_in_pref_hab_day = []
    fish_perc_in_pref_hab_night = []
    fish_perc_in_2pref_hab = []
    for fish_nr in range(14):
        # pref_hab = np.argmax(rel_sep_fish_counts_in_habitat[:, fish_nr, :], axis= 1)[np.max(rel_sep_fish_counts_in_habitat[:, fish_nr, :], axis= 1) != 0]
        pref_hab = np.argsort(rel_sep_fish_counts_in_habitat[:, fish_nr, :], axis= 1)[np.max(rel_sep_fish_counts_in_habitat[:, fish_nr, :], axis= 1) != 0][:, -1]
        pref_hab2 = np.argsort(rel_sep_fish_counts_in_habitat[:, fish_nr, :], axis= 1)[np.max(rel_sep_fish_counts_in_habitat[:, fish_nr, :], axis= 1) != 0][:, -2]
        fish_pref_hab.append(pref_hab)
        fish_2pref_hab.append(pref_hab2)
        dn_mask = np.arange(len(dn_borders)-1)[np.max(rel_sep_fish_counts_in_habitat[:, fish_nr, :], axis= 1) != 0]
        fish_dn_mask.append(dn_mask)

        perc_in_pref_hab = rel_sep_fish_counts_in_habitat[dn_mask, fish_nr, pref_hab]
        perc_in_pref_hab_day = rel_sep_fish_counts_in_habitat[dn_mask[dn_mask % 2 != 0], fish_nr, pref_hab[dn_mask % 2 != 0]]
        perc_in_pref_hab_night = rel_sep_fish_counts_in_habitat[dn_mask[dn_mask % 2 == 0], fish_nr, pref_hab[dn_mask % 2 == 0]]

        perc_in_2pref_hab = rel_sep_fish_counts_in_habitat[dn_mask, fish_nr, pref_hab2]

        fish_perc_in_pref_hab.append(perc_in_pref_hab)
        fish_perc_in_pref_hab_day.append(perc_in_pref_hab_day)
        fish_perc_in_pref_hab_night.append(perc_in_pref_hab_night)
        fish_perc_in_2pref_hab.append(perc_in_2pref_hab)



    bin_t_day = (np.array(bin_t) - 20*60) % (24*60*60)
    bin_t_day_mask = np.arange(len(bin_t_day), dtype=int)[bin_t_day > 12*60*60]
    bin_t_night_mask = np.arange(len(bin_t_day), dtype=int)[bin_t_day < 12*60*60]

    n_pref_hab_changes_day = []
    n_pref_hab_changes_night = []

    for fish_nr in range(len(fish_pref_hab)):
        Cday_pref_hab = fish_pref_hab[fish_nr][fish_dn_mask[fish_nr] % 2 != 0]
        Cnight_pref_hab = fish_pref_hab[fish_nr][fish_dn_mask[fish_nr] % 2 == 0]

        n_pref_hab_changes_night.append(len(np.diff(Cnight_pref_hab)[np.diff(Cnight_pref_hab) != 0]) / (len(Cnight_pref_hab)-1))
        n_pref_hab_changes_day.append(len(np.diff(Cday_pref_hab)[np.diff(Cday_pref_hab) != 0]) / (len(Cday_pref_hab)-1))



    # fig, ax = plt.subplots(7, 2, facecolor='white', figsize=(20/2.54, 36/2.54), sharex=True, sharey=True)
    # ax = np.hstack(ax)
    # for fish_nr in range(len(d_transition_times)):
    #     n, bins = np.histogram(day_d_transition_times[fish_nr], bins= np.arange(np.percentile(day_d_transition_times[fish_nr], 5), np.percentile(day_d_transition_times[fish_nr], 95), 1))
    #     n = n / np.sum(n) / (bins[1] - bins[0])
    #     bc = bins[:-1] + (bins[1] - bins[0]) / 2
    #
    #     fit, _ = curve_fit(efunc, bc[5:], n[5:])
    #
    #     ax[fish_nr].bar(bins[:-1] + (bins[1] - bins[0]) / 2, n, width = (bins[1] - bins[0]) * 0.8, color='red', alpha = 0.4)
    #     ax[fish_nr].plot(bc, efunc(bc, *fit), color='red')
    #
    #     n, bins = np.histogram(night_d_transition_times[fish_nr], bins= np.arange(np.percentile(night_d_transition_times[fish_nr], 5), np.percentile(night_d_transition_times[fish_nr], 95), 1))
    #     n = n / np.sum(n) / (bins[1] - bins[0])
    #     bc = bins[:-1] + (bins[1] - bins[0]) / 2
    #
    #     fit, _ = curve_fit(efunc, bc[5:], n[5:])
    #
    #     ax[fish_nr].bar(bins[:-1] + (bins[1] - bins[0]) / 2, n, width = (bins[1] - bins[0]) * 0.8, color='blue', alpha = 0.4)
    #     ax[fish_nr].plot(bc, efunc(bc, *fit), color='blue')
    #     # ax[fish_nr].plot(bins[:-1] + (bins[1] - bins[0]) / 2, n, color='blue')
    #     ax[fish_nr].set_yscale('log')
    #     ax[fish_nr].set_xlim([1, 200])

    # fig, ax = plt.subplots(1, 2, facecolor='white', figsize=(20/2.54, 12/2.54), sharex=True, sharey=True)
    # for fish_nr in range(len(d_transition_times)):
    #     c = np.random.rand(3)
    #     n, bins = np.histogram(day_d_transition_times[fish_nr], bins= np.arange(np.percentile(day_d_transition_times[fish_nr], 5), np.percentile(day_d_transition_times[fish_nr], 95), 1))
    #     n = n / np.sum(n) / (bins[1] - bins[0])
    #     # ax.bar(bins[:-1] + (bins[1] - bins[0]) / 2, n, width = (bins[1] - bins[0]) * 0.8, color='red', alpha = 0.4)
    #     ax[0].plot(bins[:-1] + (bins[1] - bins[0]) / 2, n, color=c)
    #
    #     n, bins = np.histogram(night_d_transition_times[fish_nr], bins= np.arange(np.percentile(night_d_transition_times[fish_nr], 5), np.percentile(night_d_transition_times[fish_nr], 95), 1))
    #     n = n / np.sum(n) / (bins[1] - bins[0])
    #     # ax.bar(bins[:-1] + (bins[1] - bins[0]) / 2, n, width = (bins[1] - bins[0]) * 0.8, color='blue', alpha = 0.4)
    #     ax[1].plot(bins[:-1] + (bins[1] - bins[0]) / 2, n, color=c)


    # col = colors[:7]
    # fig, ax = plt.subplots(figsize=(20/2.54, 12/2.54), facecolor='white')
    # ax.set_xlabel('day transition count')
    # ax.set_ylabel('night transition count')
    # for fish_nr in range(len(dn_hab_changes)):
    #     c = colors[fish_nr % 7]
    #     if fish_nr < len(col):
    #         m = 'o'
    #     else:
    #         m = 'D'
    #     # ax.plot(dn_hab_changes[fish_nr][1::2], dn_hab_changes[fish_nr][::2], 'o', color=c)
    #     ax.plot(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)],
    #             dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)], m, color=c, markersize=6)
    # ax.plot([0, 6000], [0, 6000], 'k-', lw=2)
    #
    # ### prefered habitat sep day
    # # rel_sep_fish_counts_in_habitat[dn_idx][fish_nr][hab_idx]
    #
    # fig, ax = plt.subplots(figsize=(20/2.54, 12/2.54), facecolor='white', sharex=True)
    # bp_day = ax.boxplot(fish_perc_in_pref_hab_day, positions = np.arange(len(fish_perc_in_pref_hab_day))*4, sym = '', widths=0.8)
    # bp_night = ax.boxplot(fish_perc_in_pref_hab_night, positions = np.arange(len(fish_perc_in_pref_hab_day))*4 +1, sym = '', widths=0.8, patch_artist=True)
    #
    # for patch in bp_night['boxes']:
    #     patch.set(facecolor='lightgrey')
    # for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    #     plt.setp(bp_night[element], color='k')
    #     plt.setp(bp_day[element], color='k')
    #
    # ax.set_xticks(np.arange(14)*4 + .5)
    # ax.set_xticklabels(np.arange(14) + 1)
    # #
    # ax.set_xlim([-1, 54])
    # ax.set_ylim([0, 1])
    # plt.show()

    # fig, ax = plt.subplots(2, 1, figsize=(20/2.54, 12/2.54), facecolor='white', sharex=True)
    # for fish_nr in range(len(fish_pref_hab)):
    #     c = colors[fish_nr]
    #     ax[0].plot(fish_dn_mask[fish_nr][fish_dn_mask[fish_nr] % 2 == 0], fish_perc_in_pref_hab[fish_nr][fish_dn_mask[fish_nr] % 2 == 0], color=c, marker = '.') # night cases
    #     ax[1].plot((fish_dn_mask[fish_nr][fish_dn_mask[fish_nr] % 2 != 0]), fish_perc_in_pref_hab[fish_nr][fish_dn_mask[fish_nr] % 2 != 0], color=c, marker='.') # day cases
    # ax[1].set_xlabel('Night/Day no.')
    # ax[0].set_ylabel('occupation of\nfavourite habitat')
    # ax[1].set_ylabel('occupation of\nfavourite habitat')
    # ax[0].set_ylim([0, 1])
    # ax[1].set_ylim([0, 1])
    #
    # fig, ax = plt.subplots(2, 1, figsize=(20/2.54, 12/2.54), facecolor='white', sharex=True)
    # for fish_nr in range(len(fish_pref_hab)):
    #     c = colors[fish_nr]
    #     ax[0].plot(fish_dn_mask[fish_nr][fish_dn_mask[fish_nr] % 2 == 0],
    #                fish_perc_in_pref_hab[fish_nr][fish_dn_mask[fish_nr] % 2 == 0] + fish_perc_in_2pref_hab[fish_nr][fish_dn_mask[fish_nr] % 2 == 0], color=c, marker = '.') # night cases
    #     ax[1].plot((fish_dn_mask[fish_nr][fish_dn_mask[fish_nr] % 2 != 0]),
    #                fish_perc_in_pref_hab[fish_nr][fish_dn_mask[fish_nr] % 2 != 0] + fish_perc_in_2pref_hab[fish_nr][fish_dn_mask[fish_nr] % 2 != 0], color=c, marker='.') # day cases
    # ax[1].set_xlabel('Night/Day no.')
    # ax[0].set_ylabel('occupation of\ntwo favourite habitat')
    # ax[1].set_ylabel('occupation of\ntwo favourite habitat')
    # ax[0].set_ylim([0, 1])
    # ax[1].set_ylim([0, 1])
    #
    #
    #
    # fig, ax = plt.subplots(2, 1, figsize=(20/2.54, 12/2.54), facecolor='white', sharex=True)
    # ax[0].plot(np.arange(14), n_pref_hab_changes_night)
    # ax[1].plot(np.arange(14), n_pref_hab_changes_day)
    # ax[1].set_xlabel('Fish ID')
    # ax[1].set_ylabel('rel. pref habitat changes')
    # ax[0].set_ylabel('rel. pref habitat changes')
    # ax[0].set_ylim([0, 1])
    # ax[1].set_ylim([0, 1])
    #
    # plt.figure()
    # plt.plot(n_pref_hab_changes_day, n_pref_hab_changes_night, 'o')
    # plt.xlabel('day changes')
    # plt.ylabel('night changes')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])

    # plt.show()

    mdhc = np.array(mdhc)
    mnhc = np.array(mnhc)
    sdhc = np.array(sdhc)
    snhc = np.array(snhc)

    n_pref_hab_changes_day = np.array(n_pref_hab_changes_day)
    n_pref_hab_changes_night = np.array(n_pref_hab_changes_night)

    m_mask = np.argsort(n_pref_hab_changes_day[:6])
    f_mask = np.argsort(n_pref_hab_changes_day[6:])

    #### FIGURE 1 #########################################################################################
    fig = plt.figure(facecolor='white', figsize=(18/2.54, 12/2.54))
    fs = 10
    import matplotlib.image as implt


    ax0 = fig.add_axes([2.5/18, 6/10, 12.5/18, 3.5/10])

    fig.text(.75/18, 9.5/10, 'C', ha='center', va='center', fontsize=fs+6)

    fig.text(1.375/18, 7.75/10, 'EOD frequency [Hz]', ha='center', va='center', fontsize=fs, rotation =90)

    # ax0.set_ylabel('EOD frequency [Hz]', fontsize=fs)
    fish_freqs = create_plot(datafile, shift, fish_nr_in_rec, colors, dn_borders, last_time=times[-1], ax = ax0)
    for ns, ne in zip(dn_borders[::2], dn_borders[1::2]):
        ax0.fill_between([ns, ne], [650, 650], [970, 970], color='#888888')
    ax0.set_yticks([700, 800, 900])

    ax0.set_xlim([dn_borders[0], dn_borders[dn_borders < times[-1]][-1]])
    ax0.set_ylim([650, 970])

    ax0a = fig.add_axes([16/18, 6/10, .5/18, 3.5/10])
    fig.text(15.5 / 18, 9.5 /10, 'D', ha='center', va='center', fontsize=fs + 6)

    ax0a.set_ylim([650, 970])
    ax0a.set_xlim([0, 2])
    ax0a.fill_between([0, 2], [750, 750], [970, 970], color='firebrick')
    ax0a.fill_between([0, 2], [750, 750], [650, 650], color=colors[2])

    ax0a.text(1, 860, 'male', fontsize=fs-2, color='k', va='center', ha='center', clip_on=False, rotation=90)
    ax0a.text(1, 700, 'female', fontsize=fs-2, color='k', va='center', ha='center', clip_on=False, rotation=90)


    ax0a.axis('off')

    ax1_0 = fig.add_axes([2.5/18, 3.25/10, 12.5/18, 1.75/10])
    ax1_0a = fig.add_axes([16/18, 3.25/10, .5/18, 1.75/10])
    fig.text(15.5 / 18, 5 /10, 'F', ha='center', va='center', fontsize=fs + 6)
    fig.text(.75 / 18, 5 /10, 'E', ha='center', va='center', fontsize=fs + 6)

    fig.text(1.375/18, 3.25/10, 'fraction of fish in habitat', ha='center', va='center', fontsize=fs, rotation =90)
    ax1_1 = fig.add_axes([2.5/18, 1.5/10, 12.5/18, 1.75/10])
    ax1_1a = fig.add_axes([16 / 18, 1.5 /10, .5 / 18, 1.75 /10])

    fishcount_night = np.round(np.nansum(np.array(binned_hab_fish_count)[:, bin_t_night_mask], axis = 0))
    fishcount_night[fishcount_night % 2 != 0] += 1
    fishcount_day = np.round(np.nansum(np.array(binned_hab_fish_count)[:, bin_t_day_mask], axis = 0))
    fishcount_day[fishcount_day % 2 != 0] += 1

    for hab_nr in range(len(hab_fish_count)):
        if hab_nr == 0:
            Cx = np.array(bin_t)[bin_t_day_mask]
            Cx = np.hstack([Cx[0] - 6 * 60 * 60, Cx, Cx[-1] + 6 * 60 * 60])

            Cy1 = np.zeros(len(bin_t_day_mask) + 2)

            Cy2 = np.nansum(np.array(binned_hab_fish_count)[:hab_nr + 1, bin_t_day_mask], axis=0) / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_day_mask], axis=0)
            Cy2 = np.hstack([Cy2[0], Cy2, Cy2[-1]])

            ax1_0.fill_between(Cx, Cy1, Cy2, color=hab_colors[hab_nr], zorder = 1)


            Cx = np.array(bin_t)[bin_t_night_mask]
            Cx = np.hstack([Cx[0] - 6 * 60 * 60, Cx, Cx[-1] + 6 * 60 * 60])

            Cy1 = np.zeros(len(bin_t_night_mask) + 2)

            Cy2 = np.nansum(np.array(binned_hab_fish_count)[:hab_nr + 1, bin_t_night_mask], axis=0) / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_night_mask], axis=0)
            Cy2 = np.hstack([Cy2[0], Cy2, Cy2[-1]])

            ax1_1.fill_between(Cx, Cy1, Cy2, color=hab_colors[hab_nr], zorder = 1)

        else:
            Cx = np.array(bin_t)[bin_t_day_mask]
            Cx = np.hstack([Cx[0] - 6 * 60 * 60, Cx, Cx[-1] + 6 * 60 * 60])

            Cy1 = np.nansum(np.array(binned_hab_fish_count)[:hab_nr, bin_t_day_mask], axis = 0) / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_day_mask], axis = 0)
            Cy1 = np.hstack([Cy1[0], Cy1, Cy1[-1]])

            Cy2 = np.nansum(np.array(binned_hab_fish_count)[:hab_nr + 1, bin_t_day_mask], axis=0) / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_day_mask], axis=0)
            Cy2 = np.hstack([Cy2[0], Cy2, Cy2[-1]])

            ax1_0.fill_between(Cx, Cy1, Cy2, color=hab_colors[hab_nr], zorder = 1)

            # ax1_0.fill_between(np.array(bin_t)[bin_t_day_mask],
            #                    np.nansum(np.array(binned_hab_fish_count)[:hab_nr, bin_t_day_mask], axis = 0) / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_day_mask], axis = 0),
            #                    np.nansum(np.array(binned_hab_fish_count)[:hab_nr + 1, bin_t_day_mask], axis=0) / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_day_mask], axis=0),
            #                    color=hab_colors[hab_nr])

            Cx = np.array(bin_t)[bin_t_night_mask]
            Cx = np.hstack([Cx[0] - 6 * 60 * 60, Cx, Cx[-1] + 6 * 60 * 60])

            Cy1 = np.nansum(np.array(binned_hab_fish_count)[:hab_nr, bin_t_night_mask], axis = 0) / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_night_mask], axis = 0)
            Cy1 = np.hstack([Cy1[0], Cy1, Cy1[-1]])

            Cy2 = np.nansum(np.array(binned_hab_fish_count)[:hab_nr + 1, bin_t_night_mask], axis=0) / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_night_mask], axis=0)
            Cy2 = np.hstack([Cy2[0], Cy2, Cy2[-1]])

            ax1_1.fill_between(Cx, Cy1, Cy2, color=hab_colors[hab_nr], zorder = 1)

            # ax1_1.fill_between(np.array(bin_t)[bin_t_night_mask],
            #                    np.nansum(np.array(binned_hab_fish_count)[:hab_nr, bin_t_night_mask], axis = 0) / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_night_mask], axis = 0),
            #                    np.nansum(np.array(binned_hab_fish_count)[:hab_nr + 1, bin_t_night_mask], axis=0) / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_night_mask], axis=0),
            #                    color=hab_colors[hab_nr])


    if True:
        print('\n\n### figure 1 B ###')
        print('Spearman fishcount vs rel. count in habitat\n')
        for hab_nr in range(len(hab_fish_count)):
            C_day_occupation = np.array(binned_hab_fish_count)[hab_nr, bin_t_day_mask] / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_day_mask], axis = 0)

            rho, p = scp.spearmanr(C_day_occupation, fishcount_day)
            print('DAY:   Habitat %.0f; rho:%.3f; p:%.4f, SD = %.3f, delta = %.3f' % (hab_nr, rho, p, np.std(C_day_occupation), C_day_occupation[-1] - C_day_occupation[0]))
        print('')
        for hab_nr in range(len(hab_fish_count)):
            C_night_occupation = np.array(binned_hab_fish_count)[hab_nr, bin_t_night_mask] / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_night_mask], axis = 0)
            rho, p = scp.spearmanr(C_night_occupation, fishcount_night)
            print('NIGHT: Habitat %.0f; rho:%.3f; p:%.4f, SD = %.3f, delta = %.3f' % (hab_nr, rho, p, np.std(C_night_occupation), C_night_occupation[-1] - C_night_occupation[0]))
        print('##################')
    # ax0.set_xlim([times[0], times[-1]])

    total_day_occupation = np.sum(np.array(binned_hab_fish_count)[:, bin_t_day_mask], axis=1) / np.sum(np.array(binned_hab_fish_count)[:, bin_t_day_mask])
    total_night_occupation = np.sum(np.array(binned_hab_fish_count)[:, bin_t_night_mask], axis=1) / np.sum(np.array(binned_hab_fish_count)[:, bin_t_night_mask])

    for i in range(len(total_day_occupation)):
        if i == 0:
            ax1_0a.bar(.5, total_day_occupation[i], color = hab_colors[i], width=1, zorder=1)
            ax1_1a.bar(.5, total_night_occupation[i], color = hab_colors[i], width=1, zorder=1)
        else:
            ax1_0a.bar(.5, total_day_occupation[i], bottom = np.sum(total_day_occupation[:i]), color = hab_colors[i], width=1, zorder=1)
            ax1_1a.bar(.5, total_night_occupation[i], bottom = np.sum(total_night_occupation[:i]), color = hab_colors[i], width=1, zorder=1)

    ax1_0a.spines['right'].set_visible(False)
    ax1_0a.spines['left'].set_visible(False)
    ax1_0a.spines['top'].set_visible(False)
    ax1_0a.spines['bottom'].set_visible(False)
    ax1_0a.set_xticks([])
    ax1_0a.set_yticks([])

    ax1_1a.spines['right'].set_visible(False)
    ax1_1a.spines['left'].set_visible(False)
    ax1_1a.spines['top'].set_visible(False)
    ax1_1a.spines['bottom'].set_visible(False)
    ax1_1a.set_xticks([.5])
    ax1_1a.set_xticklabels(['average'], fontsize=fs, rotation =45)
    ax1_1a.set_yticks([])


    # ax1_0a.axis('off')
    # ax1_1a.axis('off')


    ax1_0a.plot([0, 1], [0, 0], color='white', lw = .5, clip_on=False, zorder = 2)
    ax1_1a.plot([0, 1], [0, 0], color='white', lw = .5, clip_on=False, zorder = 2)
    ax1_0a.set_ylim([0, 1])
    ax1_0a.set_xlim([0, 1])
    ax1_1a.set_ylim([0, 1])
    ax1_1a.set_xlim([0, 1])
    ax1_1a.invert_yaxis()

    # ax1_0.set_xlim([times[0], times[-1]])
    ax1_0.set_xlim([dn_borders[0], dn_borders[dn_borders < times[-1]][-1]])
    ax1_0.set_xticks([])
    # ax1_1.set_xlim([times[0], times[-1]])
    ax1_1.set_xlim([dn_borders[0], dn_borders[dn_borders < times[-1]][-1]])

    time_ticks = np.arange(110 * 60 + 18 * 60 * 60, times[-1], 24*60*60)

    ax1_0.plot([time_ticks[0] - 18 * 60 * 60, time_ticks[-1] + 6 * 60 * 60], [0, 0], color='white', lw = .5, clip_on=False, zorder = 2)
    ax1_1.plot([time_ticks[0] - 18 * 60 * 60, time_ticks[-1] + 6 * 60 * 60], [0, 0], color='white', lw = .5, clip_on=False, zorder = 2)

    ax1_0.text(time_ticks[0] - 12 * 60 * 60, .5, 'day', fontsize=fs, va='center', ha='center', rotation=90)
    ax1_1.text(time_ticks[-1], .5, 'night', fontsize=fs, va='center', ha='center', rotation=90)

    ax1_0.text(time_ticks[1], 0.1, 'stacked stones', fontsize=fs-2, color='white', ha='center', va='center')
    ax1_0.text(time_ticks[1], 0.6, 'plants', fontsize=fs-2, color='k', ha='center', va='center')

    ax1_1.text(time_ticks[1], 0.825, 'open water', fontsize=fs-2, color='k', ha='center', va='center')
    ax1_1.text(time_ticks[6], 0.75, 'gravel', fontsize=fs-2, color='k', ha='center', va='center')
    ax1_1.text(time_ticks[6], 0.3, 'isolated stones', fontsize=fs-2, color='k', ha='center', va='center')


    ax0.set_xticks(time_ticks)
    # ax0.set_xticklabels(['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', 'Day 10'])
    ax0.set_xticklabels([])
    ax1_1.set_xticks(time_ticks)
    ax1_1.set_xticklabels(['day 1', 'day 2', 'day 3', 'day 4', 'day 5', 'day 6', 'day 7', 'day 8', 'day 9', 'day 10'])

    ax1_0.set_yticks([0, .5, 1])
    ax1_0.set_yticklabels(['0', '.5', '1'])
    ax1_1.set_yticks([.5, 1])
    ax1_1.set_yticklabels(['.5', '1'])
    ax1_0.set_ylim([0, 1])
    ax1_1.set_ylim([0, 1])


    ax1_0.spines['bottom'].set_visible(False)
    ax1_1.spines['top'].set_visible(False)
    ax1_0.set_xticks([])
    ax1_1.invert_yaxis()

    for ax in [ax0, ax1_0, ax1_1]:
        ax.tick_params(labelsize=fs-1)

    fig.savefig('/home/raab/paper_create/2019raab_habitats/traces_distribution.jpg', dpi=300)


    ##########################################
    # figure 2 new !!!
    fig = plt.figure(facecolor='white', figsize=(18/2.54, 17/2.54))


    # Only show ticks on the left and bottom spines


    #################
    # ax2_0 = fig.add_axes([0.1, 8 / 21, .85, 2 / 21])
    # ax2_0 = fig.add_axes([2/18., (1.25 + 13.5/4 + .75 + 13.5/8)/18, 15/18, (13.5/8) / 18])

    # ax2_0 = fig.add_axes([2.5/18, 7/18, 14/18, 1.5/18])
    ax2_0 = fig.add_axes([2.5/18, (8.75 + 5)/17, 14/18, 1.75/17])

    fig.text(.75 / 18, (10.5 + 5) / 17, 'A', ha='center', va='center', fontsize=fs + 6)

    fig.text(1.375/18, (8.75 + 5)/17, 'rel. time in habitat', ha='center', va='center', fontsize=fs, rotation=90)

    # ax2_1 = fig.add_axes([0.1, 6 / 21, .85, 2 / 21])
    # ax2_1 = fig.add_axes([2/18., (1.25 + 13.5/4 + .75)/18, 15/18, (13.5/8) / 18])

    # ax2_1 = fig.add_axes([2.5/18, 5.5/18, 14/18, 1.5/18])
    ax2_1 = fig.add_axes([2.5/18, (7 + 5)/17, 14/18, 1.75/17])

    ax2_1.fill_between([-1, 14], [0, 0], [1, 1], color='#888888')

    for enu, fish_nr in enumerate(range(len(rel_d_fish_counts_in_habitat))):
        # for enu, fish_nr in enumerate(np.hstack([m_mask, f_mask+6])):
        day_upshift = 0
        night_upshift = 0

        for hab_nr in range(len(rel_d_fish_counts_in_habitat[fish_nr])):
            ax2_0.bar(enu, rel_d_fish_counts_in_habitat[fish_nr][hab_nr], bottom=day_upshift, color=hab_colors[hab_nr], width=0.5, edgecolor='k', lw=.6)
            day_upshift += rel_d_fish_counts_in_habitat[fish_nr][hab_nr]

            ax2_1.bar(enu, rel_n_fish_counts_in_habitat[fish_nr][hab_nr], bottom=night_upshift, color=hab_colors[hab_nr], width=0.5, edgecolor='k', lw=.6)
            night_upshift += rel_n_fish_counts_in_habitat[fish_nr][hab_nr]

    print('\n\n Wixocon max occupied habitat ratio day vs night (14 va 14 values)')
    r, p = scp.wilcoxon(np.max(rel_d_fish_counts_in_habitat, axis = 1), np.max(rel_n_fish_counts_in_habitat, axis = 1))
    print('r = %.3f, p = %.3f' % (r, p))

    ax2_0.fill_between([-.25, 5.25], [1, 1], [1.2, 1.2], color='firebrick', clip_on=False, alpha=0.8)
    ax2_0.text(2.5, 1.1, 'male', fontsize=fs-2, color='k', va='center', ha='center', clip_on=False)
    ax2_0.fill_between([5.75, 13.25], [1, 1], [1.2, 1.2], color=colors[2], clip_on=False, alpha=0.8)
    ax2_0.text(9.5, 1.1, 'female', fontsize=fs-2, color='k', va='center', ha='center', clip_on=False)

    ax2_0.set_yticks([.5, 1])
    ax2_0.set_yticklabels(['.5', '1'])
    ax2_0.set_xticks([])
    ax2_1.set_xticks(np.arange(14))
    ax2_1.set_xticklabels(['', '', '', '', '', '', '', '', '', '', '', '', '', ''])

    ax2_1.set_yticks([0, .5, 1])
    ax2_1.set_yticklabels(['0', '.5', '1'])
    ax2_0.set_ylim([0, 1])
    ax2_1.set_ylim([0, 1])
    ax2_1.invert_yaxis()

    ax2_0.plot([-0.5, 13.5], [0, 0], color='white', lw = .5, clip_on=False, zorder = 2)
    ax2_1.plot([-0.5, 13.5], [0, 0], color='white', lw = .5, clip_on=False, zorder = 2)
    ax2_0.spines['bottom'].set_visible(False)
    ax2_1.spines['top'].set_visible(False)
    ax2_0.set_xticks([])

    # ax3 = fig.add_axes([.1, 3/42, .85, 4/21])
    # ax3 = fig.add_axes([2/18., 1.25/18, 15/18, (13.5/4) / 18])

    # ax3 = fig.add_axes([2.5/18, 1.5/18, 14/18, 3/18])
    ax3 = fig.add_axes([2.5/18, (2.5 + 5)/17, 14/18, 3.5/17])

    fig.text(.75 / 18, (6+5) / 17, 'B', ha='center', va='center', fontsize=fs + 6)
    fig.text(1.375/18, (4.+5)/17, 'rel. time in\npreferred habitat', ha='center', va='center', fontsize=fs, rotation=90 )

    # bp_day = ax3.boxplot(np.array(fish_perc_in_pref_hab_day)[np.hstack([m_mask, f_mask+6])], positions = np.arange(len(fish_perc_in_pref_hab_day))*4, sym = '', widths=0.7, patch_artist=True)
    bp_day = ax3.boxplot(np.array(fish_perc_in_pref_hab_day), positions = np.arange(len(fish_perc_in_pref_hab_day))*4, sym = '', widths=0.7, patch_artist=True)
    # bp_night = ax3.boxplot(np.array(fish_perc_in_pref_hab_night)[np.hstack([m_mask, f_mask+6])], positions = np.arange(len(fish_perc_in_pref_hab_day))*4 +1, sym = '', widths=0.7, patch_artist=True)
    bp_night = ax3.boxplot(np.array(fish_perc_in_pref_hab_night), positions = np.arange(len(fish_perc_in_pref_hab_day))*4 +1, sym = '', widths=0.7, patch_artist=True)

    if True:
        print('### figure 1 D ###')
        print('\n\n Mann Whitney U occupation pref. habitat day vs. night')
        for enu, fish_nr in enumerate(np.arange(len(fish_perc_in_pref_hab))):
            # for enu, fish_nr in enumerate(np.hstack([m_mask, f_mask+6])):
            u, p = scp.mannwhitneyu(fish_perc_in_pref_hab_day[fish_nr], fish_perc_in_pref_hab_night[fish_nr])
            # p *= len(np.hstack([m_mask, f_mask+6]))
            if p < 0.05:
                star = '*'
                if p < 0.01:
                    star = '**'
                if p < 0.001:
                    star = '***'
                ax3.text(enu*4 + .5, 1.1, star, va = 'center', ha='center', fontsize=fs)

            print('Fish no. %.0f: U = %.2f, p = %.3f' % (enu + 1, u, p))
        print('################')

    print('\n\n Spearmanr: ocupation of pref habitat vs EOD freq')
    oc_day = []
    oc_night = []
    freq_day = []
    freq_night = []

    for fish_nr in np.arange(6):
        oc_day.extend(fish_perc_in_pref_hab_day[fish_nr])
        freq_day.extend(np.ones(len(fish_perc_in_pref_hab_day[fish_nr])) * fish_freqs[fish_nr])

        oc_night.extend(fish_perc_in_pref_hab_night[fish_nr])
        freq_night.extend(np.ones(len(fish_perc_in_pref_hab_night[fish_nr])) * fish_freqs[fish_nr])
    r, p = scp.spearmanr(oc_day, freq_day)
    print('Male day: spearmanr: r = %.3f, p = %.3f' % (r, p))
    r, p = scp.spearmanr(oc_night, freq_night)
    print('Male night: spearmanr: r = %.3f, p = %.3f' % (r, p))

    oc_day = []
    oc_night = []
    freq_day = []
    freq_night = []

    for fish_nr in np.arange(6, 14):
        oc_day.extend(fish_perc_in_pref_hab_day[fish_nr])
        freq_day.extend(np.ones(len(fish_perc_in_pref_hab_day[fish_nr])) * fish_freqs[fish_nr])

        oc_night.extend(fish_perc_in_pref_hab_night[fish_nr])
        freq_night.extend(np.ones(len(fish_perc_in_pref_hab_night[fish_nr])) * fish_freqs[fish_nr])
    r, p = scp.spearmanr(oc_day, freq_day)
    print('Female day: spearmanr: r = %.3f, p = %.3f' % (r, p))
    r, p = scp.spearmanr(oc_night, freq_night)
    print('Female night: spearmanr: r = %.3f, p = %.3f' % (r, p))





    for patch in bp_night['boxes']:
        patch.set(facecolor='#888888')

    for patch in bp_day['boxes']:
        patch.set(facecolor='cornflowerblue')

    ax3.legend([bp_day["boxes"][0], bp_night["boxes"][0]], ['day', 'night'], loc=4, fontsize=fs-2, frameon=False, ncol=2)

    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp_night[element], color='k')
        plt.setp(bp_day[element], color='k')

    for i in range(len(colors)):
        sh = 1.75
        if i >= 9:
            sh = 2.25
        if i <= 5:
            # c = np.array(colors[:6])[m_mask][i]
            c = np.array(colors[:6])[i]
            ax3.plot(i*4+sh, -.11, 'o', color=c, markersize=6, clip_on=False)
        else:
            # c = np.array(colors[6:])[f_mask][i-6]
            c = np.array(colors[6:])[i-6]
            ax3.plot(i*4+sh, -.11, 'D', markerfacecolor='none', markeredgecolor=c, markersize=6, markeredgewidth=1.5, clip_on=False)

    ax3.set_xticks(np.arange(14)*4 + .5)
    ax3.set_xticklabels(np.arange(14) + 1)
    #
    ax3.set_xlim([-1, 54])
    ax3.set_ylim([0, 1.2])
    # ax3.set_ylabel('occupation\npref. habitat', fontsize=fs)
    ax3.set_xlabel('fish ID', fontsize=fs)
    ax3.set_yticks(np.arange(0, 1.1, .2))
    ax3.set_yticklabels(['0', '.2', '.4', '.6', '.8', '1'])

    ax2_0.set_xlim([-0.5, 13.5])
    ax2_1.set_xlim([-0.5, 13.5])

    ax4_1 = fig.add_axes([2.5/18, 2.5/17, 6 / 18, 3.5/17])
    fig.text(.75 / 18, 6 / 17, 'C', ha='center', va='center', fontsize=fs + 6)
    ax4_0 = fig.add_axes([10.5/18, 2.5/17, 6 / 18, 3.5/17])
    fig.text(9.25 / 18, 6 / 17, 'D', ha='center', va='center', fontsize=fs + 6)

    Cratio = count_m[-1] / (count_m[-1] + count_f[-1])
    ax4_0.errorbar(np.arange(5) - 0.1, np.array(list(map(lambda x: np.mean(x), np.array(ratio_day)[-1, :]))),
                    yerr=list(map(lambda x: np.std(x), np.array(ratio_day)[-1, :])), fmt='none', ecolor='k')
    # rax[0].plot(np.arange(5) - 0.1, np.array(list(map(lambda x: np.mean(x), np.array(ratio_day)[-1, :]))), 'o', color='grey')
    ax4_0.bar(np.arange(5) - 0.1, np.array(list(map(lambda x: np.mean(x), np.array(ratio_day)[-1, :]))), width=.2, color='cornflowerblue')

    ax4_0.errorbar(np.arange(5) + 0.1, np.array(list(map(lambda x: np.mean(x), np.array(ratio_night)[-1, :]))),
                    yerr=list(map(lambda x: np.std(x), np.array(ratio_night)[-1, :])), fmt='none', ecolor='k')
    # rax[0].plot(np.arange(5) + 0.1, np.array(list(map(lambda x: np.mean(x), np.array(ratio_night)[-1, :]))), 'o', color='k')
    ax4_0.bar(np.arange(5) + 0.1, np.array(list(map(lambda x: np.mean(x), np.array(ratio_night)[-1, :]))), width=.2, color='#888888')
    ax4_0.set_ylim(0, 1)
    ax4_0.set_xlim(-0.5, 4.5)

    # ax4_0.text(0, .075, u'\u2642:%.0f\n\u2640:%.0f' % (count_m[-1], count_f[-1]), fontsize=10, va='center', ha='center')

    ax4_0.set_xticks(np.arange(5))
    ax4_0.set_xticklabels(['st. stones', 'iso. stones', 'grass', 'gravel', 'water'], rotation = 45)
    ax4_0.tick_params(labelsize=9)
    ax4_0.set_ylabel('male ratio', fontsize=10)
    ax4_0.plot([-0.5, 4.5], [Cratio, Cratio], '--', lw=1, color='k')
    # ax4_0.legend(loc=1, fontsize=9, frameon=False)

    do_stats = False
    if do_stats == True:
        print('\n Mann-Whitney U male ratio day-night')
        for i, hab in zip(range(5), ['st. stones', 'iso. stones', 'plants', 'gravel', 'open water']):
            stats, p = scp.mannwhitneyu(np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i])
            d = cohans_d(np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i])
            print('%s: U = %.0f, p = %.3f, d = %.2f' % (hab, stats, p, d))
        print('\n Mann-Whitney U male ratio between habitats')
        for i, hab in zip(np.arange(5), np.array(['st. stones', 'iso. stones', 'plants', 'gravel', 'open water'])):
            mean = np.mean(np.hstack([np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i]]))
            std = np.std(np.hstack([np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i]]))
            print('%s: %.2f+-%.2f' % (hab, mean, std))

            for j, hab2 in zip(np.arange(5)[i+1:], np.array(['st. stones', 'iso. stones', 'plants', 'gravel', 'open water'])[i+1:]):
                stats , p = scp.mannwhitneyu(np.hstack([np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i]]),
                                             np.hstack([np.array(ratio_day)[-1, j], np.array(ratio_night)[-1, j]]))
                d = cohans_d(np.hstack([np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i]]),
                             np.hstack([np.array(ratio_day)[-1, j], np.array(ratio_night)[-1, j]]))

                print('%s - %s: U = %.0f, p = %.3f, d = %.2f' % (hab, hab2, stats, p, d))

        print("Cohan's d agains expected mean (6/14)")
        expected = 6/14
        for i, hab in zip(np.arange(5), np.array(['st. stones', 'iso. stones', 'plants', 'gravel', 'open water'])):
            mean = np.mean(np.hstack([np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i]]))
            std = np.std(np.hstack([np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i]]))

            t, p = scp.ttest_1samp(np.hstack([np.array(ratio_day)[-1, i], np.array(ratio_night)[-1, i]]), expected)
            print('\nall %s: ttest: t = %.2f, p = %.2f, d = %.2f' % (hab, t, p, np.abs((mean - expected) / std)))

            mean = np.mean(np.array(ratio_day)[-1, i])
            std = np.std(np.array(ratio_day)[-1, i])

            t, p = scp.ttest_1samp(np.array(ratio_day)[-1, i], expected)
            print('day %s: ttest: t = %.2f, p = %.2f, d = %.2f' % (hab, t, p, np.abs((mean - expected) / std)))

            mean = np.mean(np.array(ratio_night)[-1, i])
            std = np.std(np.array(ratio_night)[-1, i])

            t, p = scp.ttest_1samp(np.array(ratio_night)[-1, i], expected)
            print('night %s: ttest: t = %.2f, p = %.2f, d = %.2f' % (hab, t, p, np.abs((mean - expected) / std)))



    ax4_1.errorbar(np.arange(5) - .1, list(map(lambda x: np.mean(x), np.array(male_in_groupsize)[:, 5])),
                    yerr=list(map(lambda x: np.std(x), np.array(male_in_groupsize)[:, 5])), fmt='none', ecolor='k')
    # rax[1].plot(np.arange(5) - .1, list(map(lambda x: np.mean(x), np.array(male_in_groupsize)[:, 5])), 'o', color='blue')
    ax4_1.bar(np.arange(5) - .1, list(map(lambda x: np.mean(x), np.array(male_in_groupsize)[:, 5])), width=.2, color='firebrick', label=u'\u2642')

    ax4_1.errorbar(np.arange(5) + .1, list(map(lambda x: np.mean(x), np.array(female_in_groupsize)[:, 5])),
                    yerr=list(map(lambda x: np.std(x), np.array(female_in_groupsize)[:, 5])), fmt='none', ecolor='k')
    # rax[1].plot(np.arange(5) + .1, list(map(lambda x: np.mean(x), np.array(female_in_groupsize)[:, 5])), 'o', color='pink')
    ax4_1.bar(np.arange(5) + .1, list(map(lambda x: np.mean(x), np.array(female_in_groupsize)[:, 5])), width=.2, color=colors[2], label=u'\u2640')

    ax4_1.set_xticks(np.arange(5))
    ax4_1.set_xticklabels(['st. stones', 'iso. stones', 'grass', 'gravel', 'water'], rotation = 45)
    ax4_1.tick_params(labelsize=9)
    ax4_1.set_ylabel('group size', fontsize=10)
    ax4_1.legend(loc=1, fontsize=fs-2, frameon=False)

    if do_stats == True:
        print('\n Mann-Whitney U groupsize male-female')
        for i, hab in zip(range(5), ['st. stones', 'iso. stones', 'plants', 'gravel', 'open water']):
            stats, p = scp.mannwhitneyu(np.array(male_in_groupsize)[i, 5], np.array(female_in_groupsize)[i, 5])
            d = cohans_d(np.array(male_in_groupsize)[i, 5], np.array(female_in_groupsize)[i, 5])
            print('%s: U = %.0f, p = %.3f, d = %.2f' % (hab, stats, p, d))

        print('\n Mann-Whitney U groupsize between hab')
        for i, hab in zip(range(5), ['st. stones', 'iso. stones', 'plants', 'gravel', 'open water']):
            mean = np.mean(np.hstack([np.array(male_in_groupsize)[i, 5], np.array(female_in_groupsize)[i, 5]]))
            std = np.std(np.hstack([np.array(male_in_groupsize)[i, 5], np.array(female_in_groupsize)[i, 5]]))
            print('%s: %.2f+-%.2f' % (hab, mean, std))
            for j, hab2 in zip(np.arange(5)[i + 1:], np.array(['st. stones', 'iso. stones', 'plants', 'gravel', 'open water'])[i + 1:]):
                stats, p = scp.mannwhitneyu(np.hstack([np.array(male_in_groupsize)[i, 5], np.array(female_in_groupsize)[i, 5]]),
                                            np.hstack([np.array(male_in_groupsize)[j, 5], np.array(female_in_groupsize)[j, 5]]))

                d = cohans_d(np.hstack([np.array(male_in_groupsize)[i, 5], np.array(female_in_groupsize)[i, 5]]),
                             np.hstack([np.array(male_in_groupsize)[j, 5], np.array(female_in_groupsize)[j, 5]]))
                print('%s - %s: U = %.0f, p = %.3f, d = %.2f' % (hab, hab2, stats, p, d))

    for ax in [ax2_0, ax2_1, ax3, ax4_0, ax4_1]:
        ax.tick_params(labelsize=fs-1)

    fig.savefig('/home/raab/paper_create/2019raab_habitats/habitat_pref.jpg', dpi=300)


    ### FIGURE 3 ##########################################################################################

    fig = plt.figure(facecolor='white', figsize=(18/2.54, 12/2.54))

    fs = 10

    ax0_0 = fig.add_axes([2.75/18, 8/10, 14/18, 1.5/10])
    ax0_1 = fig.add_axes([2.75/18, 6/10, 14/18, 1.5/10])

    # ax0_1.errorbar(np.arange(len(mdhc))[:6], mdhc[:6], yerr=sdhc[:6], color='cornflowerblue', lw=2, elinewidth = 1.5, label='day')
    ax0_1.bar(np.arange(len(mdhc))[:6]-.125, mdhc[:6], yerr=sdhc[:6], color='cornflowerblue', width=.25, label='day')

    # ax0_1.errorbar(np.arange(len(mdhc))[6:], mdhc[6:], yerr=sdhc[6:], color='cornflowerblue', lw=2, elinewidth = 1.5)
    ax0_1.bar(np.arange(len(mdhc))[6:]-.125, mdhc[6:], yerr=sdhc[6:], color='cornflowerblue', width =.25)

    # ax0_1.errorbar(np.arange(len(mnhc))[:6], mnhc[:6], yerr=snhc[:6], color='#888888', lw=2, elinewidth = 1.5, label='night')
    ax0_1.bar(np.arange(len(mnhc))[:6]+.125, mnhc[:6], yerr=snhc[:6], color='#888888', width=.25, label='night')

    # ax0_1.errorbar(np.arange(len(mnhc))[6:], mnhc[6:], yerr=snhc[6:], color='#888888', lw=2, elinewidth = 1.5)
    ax0_1.bar(np.arange(len(mnhc))[6:]+.125, mnhc[6:], yerr=snhc[6:], color='#888888', width=.25)

    ax0_1.set_xticks([])
    ax0_1.set_yticks([0, 5000, 10000, 15000])
    ax0_1.set_yticklabels(['0', '5', '10', '15'])
    ax0_1.legend(fontsize = fs-2, frameon=False, loc = 1)

    # ax0_0.set_ylabel('transitions\n[n/12h]')
    # fig.text(1.5 / 18, (15.75 + 1.75/2) / 18, 'transitions\n[n/12h]', ha='center', va='center', fontsize=fs, rotation=90)
    fig.text(1.5 / 18, 8.75 / 10, 'preference\nchange [1/d]', ha='center', va='center', fontsize=fs, rotation=90)
    # fig.text(.75 / 18, (15.75 + 1.75) / 18, 'A', ha='center', va='center', fontsize=fs + 6)
    fig.text(.75 / 18, 9.5/ 10, 'A', ha='center', va='center', fontsize=fs + 6)

    ax0_1.set_ylim([0, 15000])

    if True:
        # MALEFEMALE
        sens_cutoff = [4, 5, 6, 7, 8]
        print('\n\n### figure 2 A ####')
        for sco in sens_cutoff:
            print('\n %.0f males // %.0f females' % (sco, 14-sco))
            print('Spearmanr habitat changes vs. frequency (all)\n')
            x = []
            f = []
            for fish_nr in np.arange(len(d_hab_changes))[:sco]:
                x.extend(d_hab_changes[fish_nr])
                f.extend(np.ones(len(d_hab_changes[fish_nr]))*fish_freqs[fish_nr])

            rho, p = scp.spearmanr(x, f)
            print('Male Day: rho = %.3f, p = %.3f' % (rho, p))
            xm1 = x
            x = []
            f = []
            for fish_nr in np.arange(len(d_hab_changes))[sco:]:
                x.extend(d_hab_changes[fish_nr])
                f.extend(np.ones(len(d_hab_changes[fish_nr]))*fish_freqs[fish_nr])
            rho, p = scp.spearmanr(x, f)
            print('Female Day: rho = %.3f, p = %.3f' % (rho, p))
            xf1 = x

            x = []
            f = []
            for fish_nr in np.arange(len(n_hab_changes))[:sco]:
                x.extend(n_hab_changes[fish_nr])
                f.extend(np.ones(len(n_hab_changes[fish_nr]))*fish_freqs[fish_nr])
            rho, p = scp.spearmanr(x, f)
            print('Male Night: rho = %.3f, p = %.3f' % (rho, p))
            xm2 = x

            x = []
            f = []
            for fish_nr in np.arange(len(n_hab_changes))[sco:]:
                x.extend(n_hab_changes[fish_nr])
                f.extend(np.ones(len(n_hab_changes[fish_nr]))*fish_freqs[fish_nr])
            rho, p = scp.spearmanr(x, f)
            print('Female Night: rho = %.3f, p = %.3f' % (rho, p))
            xf2 = x
            print('########################')

            print('\n\n ### figure 2 B ####')
            print('spearman pref. habitat changes vs. frequency\n')
            rho, p = scp.spearmanr(n_pref_hab_changes_day[:sco], fish_freqs[:sco])
            print('Day Male Day: rho = %.3f, p = %.3f' % (rho, p))
            rho, p = scp.spearmanr(n_pref_hab_changes_day[sco:], fish_freqs[sco:])
            print('Day Female Day: rho = %.3f, p = %.3f' % (rho, p))

            rho, p = scp.spearmanr(n_pref_hab_changes_night[:sco], fish_freqs[:sco])
            print('Night Male Day: rho = %.3f, p = %.3f' % (rho, p))
            rho, p = scp.spearmanr(n_pref_hab_changes_night[sco:], fish_freqs[sco:])
            print('Night Female Day: rho = %.3f, p = %.3f' % (rho, p))

            print('#########################')

            print('\n\n ### figure 2 C ####')
            # print('Wilcox rank habitat changes day vs. night (all)\n')
            # stat, p = scp.wilcoxon(xm1, xm2)
            # print('Male: stat = %.2f, p = %.3f' % (stat, p))
            # stat, p = scp.wilcoxon(xf1, xf2)
            # print('Female: stat = %.2f, p = %.3f' % (stat, p))
            print('spearman habitat changes day vs. night (all)\n')
            stat, p = scp.spearmanr(xm1, xm2)
            print('Male: rho = %.2f, p = %.3f' % (stat, p))
            stat, p = scp.spearmanr(xf1, xf2)
            print('Female: rho = %.2f, p = %.3f' % (stat, p))
            print('###########################')

            print('\n\n### figure 2 D ####')
            print('Wilcoxon: pref. change day vs night')
            stat, p = scp.wilcoxon(n_pref_hab_changes_day[:sco], n_pref_hab_changes_night[:sco])
            print('Male: stat = %.0f, p = %.3f' % (stat, p))
            stat, p = scp.wilcoxon(n_pref_hab_changes_day[sco:], n_pref_hab_changes_night[sco:])
            print('Female: stat = %.0f, p = %.3f' % (stat, p))

            print('Wilcoxon: pref. change day vs night')
            stat, p = scp.wilcoxon(n_pref_hab_changes_day, n_pref_hab_changes_night)
            print('together: stat = %.0f, p = %.3f' % (stat, p))
            print('####################')



        #
        # print('\n Spearmanr habitat changes vs. frequency (mean)\n')
        # rho, p = scp.spearmanr(mdhc[:6], fish_freqs[:6])
        # print('Male Day: rho = %.3f, p = %.3f' % (rho, p))
        # rho, p = scp.spearmanr(mdhc[6:], fish_freqs[6:])
        # print('Female Day: rho = %.3f, p = %.3f' % (rho, p))
        #
        # rho, p = scp.spearmanr(mnhc[:6], fish_freqs[:6])
        # print('Male Night: rho = %.3f, p = %.3f' % (rho, p))
        # rho, p = scp.spearmanr(mnhc[6:], fish_freqs[6:])
        # print('Female Night: rho = %.3f, p = %.3f' % (rho, p))

    # ax0 = fig.add_axes([.1, .8 / 3 * 2 + 3 * 0.05, .85, .8 / 3])
    # ax0_1 = fig.add_axes([.1, .8 / 3 * 2 + 3 * 0.05, .85, .8 / 3 * 0.45])

    # ax0_1 = fig.add_axes([2.75/18, 13.5/18, 14/18, 1.75/18])



    # ax0_0.plot(np.arange(14)[:6], n_pref_hab_changes_day[:6], color='cornflowerblue', lw = 2)
    ax0_0.bar(np.arange(14)[:6]-.125, n_pref_hab_changes_day[:6], color='cornflowerblue', width=.25)

    # ax0_0.plot(np.arange(14)[6:], n_pref_hab_changes_day[6:], color='cornflowerblue', lw = 2)
    ax0_0.bar(np.arange(14)[6:]-.125, n_pref_hab_changes_day[6:], color='cornflowerblue', width=.25)

    # ax0_0.plot(np.arange(14)[:6], n_pref_hab_changes_night[:6], color='#888888', lw = 2)
    ax0_0.bar(np.arange(14)[:6]+.125, n_pref_hab_changes_night[:6], color='#888888', width=.25)

    # ax0_0.plot(np.arange(14)[6:], n_pref_hab_changes_night[6:], color='#888888', lw = 2)
    ax0_0.bar(np.arange(14)[6:]+.125, n_pref_hab_changes_night[6:], color='#888888', width=.25)

    ax0_0.fill_between([-.25, 5.25], [1.05, 1.05], [1.25, 1.25], color='firebrick', clip_on=False, alpha=0.8)
    ax0_0.text(2.5, 1.15, 'male', fontsize=fs-2, color='k', va='center', ha='center', clip_on=False)
    ax0_0.fill_between([5.75, 13.25], [1.05, 1.05], [1.25, 1.25], color=colors[2], clip_on=False, alpha=0.8)
    ax0_0.text(9.5, 1.15, 'female', fontsize=fs-2, color='k', va='center', ha='center', clip_on=False)

    for i in range(len(colors)):
        sh = .35
        if i >= 9:
            sh = .4

        if i <= 5:
            # c = np.array(colors[:6])[m_mask][i]
            c = np.array(colors[:6])[i]
            ax0_1.plot(i+sh, -3000, 'o', color=c, markersize=6, clip_on=False)
        else:
            # c = np.array(colors[6:])[f_mask][i-6]
            c = np.array(colors[6:])[i-6]
            ax0_1.plot(i+sh, -3000, 'D', markerfacecolor='none', markeredgecolor=c, markersize=6, markeredgewidth=1.5, clip_on=False)

    # ax0_1.set_ylabel('preference\nchanges', fontsize=10)
    # fig.text(1.5 / 18, (13.5 + 1.75 / 2) / 18, 'preference\nchange', ha='center', va='center', fontsize=fs, rotation=90)
    fig.text(1.5 / 18, 6.75/10, 'transitions\n[1000/12h]', ha='center', va='center', fontsize=fs, rotation=90)
    # fig.text(.75 / 18, (13.5 + 1.75) / 18, 'B', ha='center', va='center', fontsize=fs + 6)
    fig.text(.75 / 18, 7.5/ 10, 'B', ha='center', va='center', fontsize=fs + 6)

    ax0_0.set_yticks([0, .5, 1])
    ax0_0.set_yticklabels(['0', '.5', '1'])

    ax0_0.set_xticks(np.arange(14))
    ax0_0.set_xticklabels(['', '', '', '', '', '', '', '', '', '', '', '', '', ''])
    ax0_1.set_xticks(np.arange(14))
    ax0_1.set_xticklabels(np.arange(14)+1)

    ax0_1.set_xlabel('fish ID', fontsize=fs)

    ax0_0.set_xlim([-.5, 13.5])
    ax0_1.set_xlim([-.5, 13.5])
    ax0_0.set_ylim([0, 1])

    ###################################



    # ax0 = fig.add_axes([.1, .8/3 * 2 + 3 * 0.05, .85, .8/3])
    # bp_day = ax0.boxplot(fish_perc_in_pref_hab_day, positions = np.arange(len(fish_perc_in_pref_hab_day))*4, sym = '', widths=0.8)
    # bp_night = ax0.boxplot(fish_perc_in_pref_hab_night, positions = np.arange(len(fish_perc_in_pref_hab_day))*4 +1, sym = '', widths=0.8, patch_artist=True)
    #
    # if True:
    #     print('\n Mann Whitney U occupation pref. habitat day vs. night')
    #     for fish_nr in range(len(fish_perc_in_pref_hab)):
    #         u, p = scp.mannwhitneyu(fish_perc_in_pref_hab_day[fish_nr], fish_perc_in_pref_hab_day[fish_nr])
    #         print('Fish no. %.0f: U = %.2f, p = %.3f' % (fish_nr, u, p))
    #
    #
    # for patch in bp_night['boxes']:
    #     patch.set(facecolor='lightgrey')
    # for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    #     plt.setp(bp_night[element], color='k')
    #     plt.setp(bp_day[element], color='k')
    #
    # ax0.set_xticks(np.arange(14)*4 + .5)
    # ax0.set_xticklabels(np.arange(14) + 1)
    # #
    # ax0.set_xlim([-1, 54])
    # ax0.set_ylim([0, 1])
    # ax0.set_ylabel('occupation\npref. habitat', fontsize=10)
    # ax0.set_xlabel('Fish ID', fontsize=10)
    # ax0.set_yticks(np.arange(0, 1.1, .2))
    # ax0.set_yticklabels(['0', '.2', '.4', '.6', '.8', '1'])


    #############################################################

    # ax1_0 = fig.add_axes([.1, .8/3 + .1, 0.375, .8/3])
    # ax1_0 = fig.add_axes([2.75/18, 7.5/18, 6/18, 4/18])
    ax1_0 = fig.add_axes([2.75/18, 1.25/10, 5.75/18, 3.25/10])
    ax1_1 = fig.add_axes([11.25/18, 1.25/10, 5.75/18, 3.25/10])

    col = colors[:7]
    col.append('k')
    for fish_nr in range(len(dn_hab_changes))[:6]:
        c = col[fish_nr]
        m = 'o'
        # ax.plot(dn_hab_changes[fish_nr][1::2], dn_hab_changes[fish_nr][::2], 'o', color=c)
        ax1_1.plot(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)],
                   dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)], m, color=c, markersize=3, alpha = .4)
        ax1_1.errorbar(np.mean(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                       np.mean(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                       xerr = np.std(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                       yerr = np.std(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                       color='firebrick', marker=m, markersize=4)

    for enu, fish_nr in enumerate(range(len(dn_hab_changes))[6:]):
        c = col[enu]
        m = 'D'
        # ax.plot(dn_hab_changes[fish_nr][1::2], dn_hab_changes[fish_nr][::2], 'o', color=c)
        ax1_1.plot(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)],
                   dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)], m, markerfacecolor='none', markeredgecolor=c, markersize=3, alpha = .4)
        ax1_1.errorbar(np.mean(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                       np.mean(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                       xerr = np.std(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                       yerr = np.std(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                       color = colors[2], markeredgecolor=colors[2], marker=m, markerfacecolor='none', markersize=4)

    ax1_1.plot([0, 6000], [0, 6000], 'k-', lw=1)

    ax1_1.set_xlabel('day transitions [1000/12h]', fontsize=fs)

    ax1_1.set_xlim([0, 6000])
    ax1_1.set_xticks(np.arange(0, 6001, 1000, dtype=int))
    # ax1_0.set_xticklabels(np.arange(7))
    ax1_1.set_xticklabels(['0', '1', '2', '3', '4', '5', '6'])

    # ax1_0.set_ylabel('# night transitions', fontsize=10)
    # fig.text(1.5 / 18, 9.5 / 18, 'night transitions\n[n/12h]', ha='center', va='center', fontsize=fs, rotation=90)
    # fig.text(7.5 / 18, 3 / 10, 'night transitions\n[n/12h]', ha='center', va='center', fontsize=fs, rotation=90)
    ax1_1.set_ylabel('night transitions\n[1000/12h]', fontsize=fs)
    # fig.text(.75 / 18, 11.5 / 18, 'C', ha='center', va='center', fontsize=fs + 6)
    fig.text(.75 / 18, 4.5/10, 'C', ha='center', va='center', fontsize=fs + 6)

    ax1_1.set_ylim([0, 15000])
    ax1_1.set_yticks(np.arange(0, 14001, 2000))
    # ax1_0.set_yticklabels(np.arange(0, 14.1, 2, dtype=int))
    ax1_1.set_yticklabels(['0', '2', '4', '6', '8', '10', '12', '14'])



    # ax1_1 = fig.add_axes([.575, .8/3 + .1, 0.375, .8/3])
    # ax1_1 = fig.add_axes([10.75/18, 7.5/18, 6/18, 4/18])


    for enu in range(len(colors)):
        if enu <= 5:
            ax1_0.plot(n_pref_hab_changes_day[enu], n_pref_hab_changes_night[enu], 'o', color=colors[enu], markersize=5, clip_on=False)
        else:
            ax1_0.plot(n_pref_hab_changes_day[enu], n_pref_hab_changes_night[enu], 'D', markerfacecolor='none', markeredgecolor=colors[enu], markersize=5, markeredgewidth=1.5, clip_on=False)

    # if True:
    #     print('\n Wilcoxon rank test pref. habitat changes day vs. night')
    #     stat, p = scp.wilcoxon(n_pref_hab_changes_day[:6], n_pref_hab_changes_night[:6])
    #     print('Male: stat = %.3f, p = %.3f' % (stat, p))
    #     stat, p = scp.wilcoxon(n_pref_hab_changes_day[6:], n_pref_hab_changes_night[6:])
    #     print('Female: stat = %.3f, p = %.3f' % (stat, p))
    #
    #     print('\n spearmanr pref. habitat changes day vs. night')
    #     stat, p = scp.spearmanr(n_pref_hab_changes_day[:6], n_pref_hab_changes_night[:6])
    #     print('Male: rho = %.3f, p = %.3f' % (stat, p))
    #     stat, p = scp.spearmanr(n_pref_hab_changes_day[6:], n_pref_hab_changes_night[6:])
    #     print('Female: rho = %.3f, p = %.3f' % (stat, p))

    ax1_0.plot([-0.01, 1.01], [-0.01, 1.01], color='k', lw=1)
    ax1_0.set_xlabel('day preference change [1/d]', fontsize=fs)

    fig.text(1.5 / 18, 3 / 10, 'night preference\nchange [1/d]', ha='center', va='center', fontsize=fs, rotation=90)
    # fig.text(9.25 / 18, 9.5 / 18, '# night transitions', ha='center', va='center', fontsize=fs, rotation=90)
    # fig.text(9.5 / 18, 11.5 / 18, 'D', ha='center', va='center', fontsize=fs + 6)
    fig.text(10 / 18, 4.5 / 10, 'D', ha='center', va='center', fontsize=fs + 6)

    ax1_0.set_xlim([-0.01, 1.01])
    ax1_0.set_ylim([-0.01, 1.01])
    ax1_0.set_yticks(np.arange(0, 1.1, .2))
    ax1_0.set_yticklabels(['0', '.2', '.4', '.6', '.8', '1'])
    ax1_0.set_xticks(np.arange(0, 1.1, .2))
    ax1_0.set_xticklabels(['0', '.2', '.4', '.6', '.8', '1'])

    for ax in [ax0_0, ax0_1, ax1_0, ax1_1]:
        ax.tick_params(labelsize=fs - 1)

    for ax in [ax0_0, ax0_1]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    fig.savefig('/home/raab/paper_create/2019raab_habitats/transitions.jpg', dpi=300)

    plot_for_Jan = False
    if plot_for_Jan:
        fig = plt.figure(figsize=(20/2.54, 12/2.54), facecolor='white')
        ax = fig.add_axes([.1, .15, .85, .8])
        #
        # ax = fig.add_axes([.1, .15, .35, .8])
        # ax1 = fig.add_axes([.55, .6, .4, .35])
        # ax2 = fig.add_axes([.55, .15, .4, .35])

        col = colors[:7]
        col.append('k')
        for fish_nr in range(len(dn_hab_changes))[:6]:
            m = 'o'
            if fish_nr == 0:
                ax.errorbar(np.mean(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               np.mean(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               xerr = np.std(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               yerr = np.std(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               color='dodgerblue', marker=m, markersize=5, label=u'\u2642')
            else:
                ax.errorbar(np.mean(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               np.mean(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               xerr = np.std(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               yerr = np.std(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               color='dodgerblue', marker=m, markersize=5)
        for enu, fish_nr in enumerate(range(len(dn_hab_changes))[6:]):
            m = 'o'
            if enu == 0:
                ax.errorbar(np.mean(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               np.mean(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               xerr = np.std(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               yerr = np.std(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               color = 'deeppink', marker=m, markersize=5, label = u'\u2640')
            else:
                ax.errorbar(np.mean(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               np.mean(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               xerr = np.std(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               yerr = np.std(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               color = 'deeppink', marker=m, markersize=5)
        ax.plot([0, 6000], [0, 6000], 'k-', lw=1)

        ax.set_xlabel('day transitions [1000/12h]', fontsize=fs)
        ax.set_xlim([0, 5000])
        ax.set_xticks(np.arange(0, 5001, 1000, dtype=int))
        ax.set_xticklabels(['0', '1', '2', '3', '4', '5'])

        ax.set_ylabel('night transitions\n[1000/12h]', fontsize=fs)
        ax.set_ylim([0, 14000])
        ax.set_yticks(np.arange(0, 14001, 2000))
        ax.set_yticklabels(['0', '2', '4', '6', '8', '10', '12', '14'])
        ax.legend(loc=1, frameon=False, fontsize=fs)

        ax.tick_params(labelsize=fs - 1)

        ######################################
        for fish_nr in range(len(dn_hab_changes))[:6]:
            m = 'o'
            if fish_nr == 0:
                ax1.errorbar(fish_freqs[fish_nr],
                               np.mean(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               yerr = np.std(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               color='dodgerblue', marker=m, markersize=5, label=u'\u2642')
            else:
                ax1.errorbar(fish_freqs[fish_nr],
                               np.mean(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               yerr = np.std(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               color='dodgerblue', marker=m, markersize=5)
        for enu, fish_nr in enumerate(range(len(dn_hab_changes))[6:]):
            m = 'o'
            if enu == 0:
                ax1.errorbar(fish_freqs[fish_nr],
                               np.mean(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               yerr = np.std(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               color = 'deeppink', marker=m, markersize=5, label = u'\u2640')
            else:
                ax1.errorbar(fish_freqs[fish_nr],
                               np.mean(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               yerr = np.std(dn_hab_changes[fish_nr][::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               color = 'deeppink', marker=m, markersize=5)

        ax1.set_ylabel('night transitions\n[1000/12h]', fontsize=fs)
        ax1.set_ylim([0, 14000])
        ax1.set_yticks(np.arange(0, 14001, 2000))
        ax1.set_yticklabels(['0', '2', '4', '6', '8', '10', '12', '14'])


        ################
        for fish_nr in range(len(dn_hab_changes))[:6]:
            m = 'o'
            if fish_nr == 0:
                ax2.errorbar(fish_freqs[fish_nr],
                               np.mean(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               yerr = np.std(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               color='dodgerblue', marker=m, markersize=5, label=u'\u2642')
            else:
                ax2.errorbar(fish_freqs[fish_nr],
                               np.mean(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               yerr = np.std(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               color='dodgerblue', marker=m, markersize=5)
        for enu, fish_nr in enumerate(range(len(dn_hab_changes))[6:]):
            m = 'o'
            if enu == 0:
                ax2.errorbar(fish_freqs[fish_nr],
                               np.mean(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               yerr = np.std(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               color = 'deeppink', marker=m, markersize=5, label = u'\u2640')
            else:
                ax2.errorbar(fish_freqs[fish_nr],
                               np.mean(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               yerr = np.std(dn_hab_changes[fish_nr][1::2][(dn_hab_changes[fish_nr][1::2] != 0) & (dn_hab_changes[fish_nr][::2] != 0)]),
                               color = 'deeppink', marker=m, markersize=5)

        ax2.set_ylabel('day transitions\n[1000/12h]', fontsize=fs)
        ax2.set_ylim([0, 5000])
        ax2.set_yticks(np.arange(0, 5001, 1000, dtype=int))
        ax2.set_yticklabels(['0', '1', '2', '3', '4', '5'])
        #
        ax2.set_xlabel('EOD frequency [Hz]', fontsize=fs)

    ##############################################
    fig = plt.figure(facecolor='white', figsize=(18/2.54, 8/2.54))


    # ax2_0 = fig.add_axes([2.75/18, 3.75/18, 4.75/18, 1.75/18])
    ax2_0 = fig.add_axes([2.75/18, 5/8, 3.25/18, 2.5/8])

    # n, bins = np.histogram(day_d_transition_times[1], bins= np.arange(np.percentile(day_d_transition_times[1], 5), np.percentile(day_d_transition_times[1], 95), 1))
    n, bins = np.histogram(day_d_transition_times[1], bins= np.linspace(0, np.percentile(day_d_transition_times[1], 100), 100))
    n = n / np.sum(n) / (bins[1] - bins[0])
    # ax2_0.bar(bins[:-1] + (bins[1] - bins[0]) / 2, n, width = (bins[1] - bins[0]) * 0.8, color='cornflowerblue', alpha = 0.4)
    bc = bins[:-1] + (bins[1:] - bins[:-1]) / (np.diff(bins) / 2)

    # data = day_d_transition_times[1]
    # kde = scp.gaussian_kde(data)
    # x = np.arange(0, 5000, 1)
    # n_smooth = kde(x)

    ax2_0.plot(bc, n, color='cornflowerblue')
    # ax2_0.plot(x, n_smooth, color='cornflowerblue')
    ax2_0.fill_between(bc, n, np.ones(len(n))*(10**-8), color='cornflowerblue', label='day')
    # ax2_0.fill_between(x, n_smooth, np.ones(len(n_smooth))*(10**-8), color='cornflowerblue', label='day')

    # n, bins = np.histogram(night_d_transition_times[1], bins= np.arange(np.percentile(night_d_transition_times[1], 5), np.percentile(night_d_transition_times[1], 95), 1))
    n, bins = np.histogram(night_d_transition_times[1], bins= np.linspace(0, np.percentile(day_d_transition_times[1], 100), 100))
    n = n / np.sum(n) / (bins[1] - bins[0])
    # ax2_0.bar(bins[:-1] + (bins[1] - bins[0]) / 2, n, width = (bins[1] - bins[0]) * 0.8, color='#888888', alpha = 0.4)
    bc = bins[:-1] + (bins[1:] - bins[:-1]) / (np.diff(bins) / 2)

    # data = night_d_transition_times[1]
    # kde = scp.gaussian_kde(data)
    # x = np.arange(0, 5000, 1)
    # n_smooth = kde(x)

    ax2_0.plot(bc, n, color='#888888')
    # ax2_0.plot(x, n_smooth, color='#888888')
    ax2_0.fill_between(bc, n, np.ones(len(n))*(10**-8), color='#888888', label='night')
    # ax2_0.fill_between(x, n_smooth, np.ones(len(n_smooth))*(10**-8), color='#888888', label='night')

    # ax2_0.plot(bins[:-1] + (bins[1] - bins[0]) / 2, n, color='#888888')
    ax2_0.set_yscale('log')
    ax2_0.legend(fontsize=fs-2, loc=1, frameon=False)
    # ax2_0.set_ylim([10**(-8), 10**0])

    # ax2_1 = fig.add_axes([2.75/18, 1.5/18, 4.75/18, 1.75/18])
    # ax2_1 = fig.add_axes([2.75/18, 1.5/18, 3.25/18, 1.75/18])
    ax2_1 = fig.add_axes([2.75/18, 1.5/8, 3.25/18, 2.5/8])

    # n, bins = np.histogram(day_d_transition_times[13], bins= np.arange(np.percentile(day_d_transition_times[13], 5), np.percentile(day_d_transition_times[13], 95), 1))
    n, bins = np.histogram(day_d_transition_times[13], bins= np.linspace(0, np.percentile(day_d_transition_times[1], 100), 100))
    n = n / np.sum(n) / (bins[1] - bins[0])
    # ax2_1.bar(bins[:-1] + (bins[1] - bins[0]) / 2, n, width = (bins[1] - bins[0]) * 0.8, color='cornflowerblue', alpha = 0.4)
    bc = bins[:-1] + (bins[1:] - bins[:-1]) / (np.diff(bins) / 2)

    # data = day_d_transition_times[13]
    # kde = scp.gaussian_kde(data)
    # x = np.arange(0, 5000, 1)
    # n_smooth = kde(x)

    ax2_1.plot(bc, n, color='cornflowerblue')
    # ax2_1.plot(x, n_smooth, color='cornflowerblue')
    ax2_1.fill_between(bc, n, np.ones(len(n))*(10**-8), color='cornflowerblue')
    # ax2_1.fill_between(x, n_smooth, np.ones(len(n_smooth))*(10**-8), color='cornflowerblue')

    # ax2_1.plot(bins[:-1] + (bins[1] - bins[0]) / 2, n, color='cornflowerblue')

    n, bins = np.histogram(night_d_transition_times[13], bins= np.linspace(0, np.percentile(day_d_transition_times[1], 100), 100))
    n = n / np.sum(n) / (bins[1] - bins[0])
    # ax2_1.bar(bins[:-1] + (bins[1] - bins[0]) / 2, n, width = (bins[1] - bins[0]) * 0.8, color='#888888', alpha = 0.4)
    bc = bins[:-1] + (bins[1:] - bins[:-1]) / (np.diff(bins) / 2)

    # data = night_d_transition_times[13]
    # kde = scp.gaussian_kde(data)
    # x = np.arange(0, 5000, 1)
    # n_smooth = kde(x)

    ax2_1.plot(bc, n, color='#888888')
    # ax2_1.plot(x, n_smooth, color='#888888')
    ax2_1.fill_between(bc, n, np.ones(len(n))*(10**-8), color='#888888')
    # ax2_1.fill_between(x, n_smooth, np.ones(len(n_smooth))*(10**-8), color='#888888')
    # ax2_1.plot(bins[:-1] + (bins[1] - bins[0]) / 2, n, color='#888888')

    ax2_1.set_yscale('log')

    # ax2_1.set_ylabel('probability', fontsize=10)
    # fig.text(.025, .05 + .4/3, 'probability', fontsize=10, va='center', ha='center', rotation=90)
    fig.text(1.5 / 18, 4.5 / 8, 'probability density', ha='center', va='center', fontsize=fs, rotation=90)
    fig.text(.75 / 18, 7.5 / 8, 'A', ha='center', va='center', fontsize=fs + 6)


    # ax2_0a = fig.add_axes([8/18, 3.75/18, 4.75/18, 1.75/18])
    # ax2_0a = fig.add_axes([6.5/18, 3.75/18, 3.25/18, 1.75/18])
    ax2_0a = fig.add_axes([6.5/18, 5/8, 3.25/18, 2.5/8])

    ax2_0a.set_yticklabels([])
    n, bins = np.histogram(day_d_transition_times[1], bins= np.linspace(0, np.percentile(day_d_transition_times[1], 100), 100))
    # n = n / np.sum(n) / ((bins[1:] - bins[:-1]) / np.diff(bins))
    n = n / np.sum(n) / (bins[1] - bins[0])
    # ax2_0a.bar(bins[:-1] + (bins[1:] - bins[:-1]) / (np.diff(bins) / 2), n, width = np.diff(bins) * 0.8, color='cornflowerblue', alpha = 0.4)
    bc = bins[:-1] + (bins[1:] - bins[:-1]) / (np.diff(bins) / 2)

    # data = day_d_transition_times[1]
    # kde = scp.gaussian_kde(data)
    # x = np.arange(0, 5000, 1)
    # n_smooth = kde(x)

    # ax2_0a.plot(bc[(n > 0) | (bc > 10)], n[(n > 0) | (bc > 10)], color='cornflowerblue')

    ax2_0a.plot(bc, n, color='cornflowerblue')
    # ax2_0a.plot(x, n_smooth, color='cornflowerblue')
    ax2_0a.fill_between(bc, n, np.ones(len(n))*(10**-8), color='cornflowerblue')
    # ax2_0a.fill_between(x, n_smooth, np.ones(len(n_smooth))*(10**-8), color='cornflowerblue')

    n, bins = np.histogram(night_d_transition_times[1], bins= np.linspace(0, np.percentile(day_d_transition_times[1], 100), 100))
    # n = n / np.sum(n) / ((bins[1:] - bins[:-1]) / np.diff(bins))
    n = n / np.sum(n) / (bins[1] - bins[0])
    # ax2_0a.bar(bins[:-1] + (bins[1:] - bins[:-1]) / (np.diff(bins) / 2), n, width = np.diff(bins) * 0.8, color='#888888', alpha = 0.4)
    bc = bins[:-1] + (bins[1:] - bins[:-1]) / (np.diff(bins) / 2)

    # data = night_d_transition_times[1]
    # kde = scp.gaussian_kde(data)
    # x = np.arange(0, 5000, 1)
    # n_smooth = kde(x)

    # ax2_0a.plot(bc[(n > 0) | (bc > 10)], n[(n > 0) | (bc > 10)], color='#888888')

    ax2_0a.plot(bc, n, color='#888888')
    # ax2_0a.plot(x, n_smooth, color='#888888')
    ax2_0a.fill_between(bc, n, np.ones(len(n))*(10**-8), color='#888888')
    # ax2_0a.fill_between(x, n_smooth, np.ones(len(n_smooth))*(10**-8), color='#888888')

    ax2_0a.set_yscale('log')
    ax2_0a.set_xscale('log')


    # ax2_1a = fig.add_axes([8/18, 1.5/18, 4.75/18, 1.75/18])
    # ax2_1a = fig.add_axes([6.5/18, 1.5/18, 3.25/18, 1.75/18])
    ax2_1a = fig.add_axes([6.5/18, 1.5/8, 3.25/18, 2.5/8])

    ax2_1a.set_yticklabels([])
    n, bins = np.histogram(day_d_transition_times[13], bins= np.linspace(0, np.percentile(day_d_transition_times[1], 100), 100))
    # n = n / np.sum(n) / ((bins[1:] - bins[:-1]) / np.diff(bins))
    n = n / np.sum(n) / (bins[1] - bins[0])
    # ax2_1a.bar(bins[:-1] + (bins[1:] - bins[:-1]) / (np.diff(bins) / 2), n, width = np.diff(bins) * 0.8, color='cornflowerblue', alpha = .4)
    bc = bins[:-1] + (bins[1:] - bins[:-1]) / (np.diff(bins) / 2)

    # data = day_d_transition_times[13]
    # kde = scp.gaussian_kde(data)
    # x = np.arange(0, 5000, 1)
    # n_smooth = kde(x)

    # ax2_1a.plot(bc[(n > 0) | (bc > 10)], n[(n > 0) | (bc > 10)], color='cornflowerblue')
    ax2_1a.plot(bc, n, color='cornflowerblue')
    # ax2_1a.plot(x, n_smooth, color='cornflowerblue')
    ax2_1a.fill_between(bc, n, np.ones(len(n))*(10**-8), color='cornflowerblue')
    # ax2_1a.fill_between(x, n_smooth, np.ones(len(n_smooth))*(10**-8), color='cornflowerblue')

    n, bins = np.histogram(night_d_transition_times[13], bins= np.linspace(0, np.percentile(day_d_transition_times[1], 100), 100))
    # n, bins = np.histogram(night_d_transition_times[13], bins= np.logspace(np.log10(1), np.log10(np.percentile(day_d_transition_times[1], 100)), 100))
    # n = n / np.sum(n) / ((bins[1:] - bins[:-1]) / np.diff(bins))
    n = n / np.sum(n) / (bins[1] - bins[0])
    # ax2_1a.bar(bins[:-1] + (bins[1:] - bins[:-1]) / (np.diff(bins) / 2), n, width = np.diff(bins) * 0.8, color='#888888', alpha = .4)
    bc = bins[:-1] + (bins[1:] - bins[:-1]) / (np.diff(bins) / 2)

    # data = night_d_transition_times[13]
    # kde = scp.gaussian_kde(data)
    # x = np.arange(0, 5000, 1)
    # n_smooth = kde(x)

    # ax2_1a.plot(bc[(n > 0) | (bc > 10)], n[(n > 0) | (bc > 10)], color='#888888')
    ax2_1a.plot(bc, n, color='#888888')
    # ax2_1a.plot(x, n_smooth, color='#888888')
    ax2_1a.fill_between(bc, n, np.ones(len(n))*(10**-8), color='#888888')
    # ax2_1a.fill_between(x, n_smooth, np.ones(len(n_smooth))*(10**-8), color='#888888')

    ax2_1a.set_yscale('log')
    ax2_1a.set_xscale('log')

    # ax2_1 = fig.add_axes([2.75/18, 1.5/8, 3.25/18, 2.5/8])
    fig.text((6.5  + (3.25/2)) / 18, .5 / 8, 'transition times [s]', fontsize=fs, va='center', ha='center')
    fig.text((2.75 + (3.25/2)) / 18, .5 / 8, 'transition times [s]', fontsize=fs, va='center', ha='center')
    # ax2_1.set_xlabel('transition times [s]', fontsize=fs)
    # ax2_1a.set_xlabel('transition times [s]', fontsize=fs)

    ax2_0.set_ylim(ymin=10**-8, ymax= 10**-2)
    ax2_1.set_ylim(ymin=10**-8, ymax= 10**-2)
    ax2_0a.set_ylim(ymin=10**-8, ymax= 10**-2)
    ax2_1a.set_ylim(ymin=10**-8, ymax= 10**-2)

    ax2_0.set_xticklabels([])
    ax2_0a.set_xticklabels([])


    ax2_0.set_yticks([10**-8,10**-6, 10**-4, 10**-2])
    ax2_1.set_yticks([10**-8, 10**-6, 10**-4, 10**-2])
    ax2_0a.set_yticks([10**-8,10**-6, 10**-4, 10**-2])
    ax2_0a.set_yticklabels([])
    ax2_1a.set_yticks([10**-8, 10**-6, 10**-4, 10**-2])
    ax2_1a.set_yticklabels([])

    ax2_0.set_xlim([0, 5000])
    ax2_1.set_xlim([0, 5000])

    ax2_0a.set_xlim(xmin=np.log10(.1), xmax=10**4)
    ax2_1a.set_xlim(xmin=np.log10(.1), xmax=10**4)


    # ax2_2 = fig.add_axes([14.75/18, 1.5/18, 2/18, 4/18])
    # ax2_2 = fig.add_axes([11.75/18, 1.5/18, 1.5/18, 4/18])
    ax2_2 = fig.add_axes([11.75/18, 1.5/8, 1.5/18, 6/8])

    night_tau = []
    day_tau = []

    for fish_nr in range(len(d_transition_times)):
        n, bins = np.histogram(day_d_transition_times[fish_nr], bins= np.arange(np.percentile(day_d_transition_times[fish_nr], 5), np.percentile(day_d_transition_times[fish_nr], 95), 1))
        n = n / np.sum(n) / (bins[1] - bins[0])
        bc = bins[:-1] + (bins[1] - bins[0]) / 2

        fit, _ = curve_fit(efunc, bc[5:], n[5:])
        _, tau = scp.expon.fit(day_d_transition_times[fish_nr], floc=0)
        day_tau.append(np.mean(day_d_transition_times[fish_nr]))

        n, bins = np.histogram(night_d_transition_times[fish_nr], bins= np.arange(np.percentile(night_d_transition_times[fish_nr], 5), np.percentile(night_d_transition_times[fish_nr], 95), 1))
        n = n / np.sum(n) / (bins[1] - bins[0])
        bc = bins[:-1] + (bins[1] - bins[0]) / 2
        fit, _ = curve_fit(efunc, bc[5:], n[5:])
        _, tau = scp.expon.fit(night_d_transition_times[fish_nr], floc=0)
        night_tau.append(np.mean(night_d_transition_times[fish_nr]))


    bp = ax2_2.boxplot([1/ np.array(day_tau[:6]), 1/np.array(night_tau[:6]), [], 1/np.array(day_tau[6:]), 1/np.array(night_tau[6:])], sym='', patch_artist=True)
    for enu, patch in enumerate(bp['boxes']):
        if enu in [1, 4]:
            patch.set(facecolor='#888888')
        elif enu in [0, 3]:
            patch.set(facecolor='cornflowerblue')
        else:
            pass
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='k')

    ax2_2.set_xticks([1.5, 4.5])
    ax2_2.set_xticklabels(['male', 'female'], rotation=45)

    ax2_2.text(1.5, .3125, '**', va = 'center', ha='center', fontsize=fs)
    ax2_2.text(4.5, .3125, '*', va = 'center', ha='center', fontsize=fs)


    if True:
        # MALEFEMALE
        print('\n\n ### figure 2 F ####')
        # sens_cutoff = [4, 5, 6, 7, 8]
        sens_cutoff = [6]
        for sco in sens_cutoff:
            print('\n %.0f males // %.0f females' % (sco, 14 - sco))
            print('\n Wilcoxon transition rate day / night')
            stat, p = scp.mannwhitneyu(1/np.array(day_tau[:sco]), 1/np.array(night_tau[:sco]))
            d = cohans_d(1/np.array(day_tau[:sco]), 1/np.array(night_tau[:sco]))
            print('Male: U = %.3f, p = %.3f, d = %.2f' % (stat, p*2, d))
            stat, p = scp.mannwhitneyu(1/np.array(day_tau[sco:]), 1/np.array(night_tau[sco:]))
            d = cohans_d(1 / np.array(day_tau[sco:]), 1 / np.array(night_tau[sco:]))
            print('Female: U = %.3f, p = %.3f, d = %.2f' % (stat, p*2, d))

            U, p = scp.mannwhitneyu(1/np.array(day_tau[:sco]), 1/np.array(day_tau[sco:]))
            d = cohans_d(1 / np.array(day_tau[:sco]), 1 / np.array(day_tau[sco:]))
            print('Day: U = %.3f, p = %.3f, d = %.2f' %(U, p*2, d))
            U, p = scp.mannwhitneyu(1/np.array(night_tau[:sco]), 1/np.array(night_tau[sco:]))
            d = cohans_d(1 / np.array(night_tau[:sco]), 1 / np.array(night_tau[sco:]))
            print('night: U = %.3f, p = %.3f, d = %.2f' %(U, p*2, d))
            print('#########################')


    # ax2_2.set_xticks([1, 2])
    # ax2_2.set_xticklabels(['day', 'night'])
    ax2_2.set_ylabel('transition rate [Hz]', fontsize=fs)
    # fig.text(10.25 / 18, 5.5 / 18, 'F', ha='center', va='center', fontsize=fs + 6)
    fig.text(10.25 / 18, 7.5 / 8, 'B', ha='center', va='center', fontsize=fs + 6)

    ax2_2.set_ylim([0, 0.35])
    ax2_2.set_yticks(np.arange(0, 0.36, 0.1))
    ax2_2.set_yticklabels(['0', '.10', '.20', '.30'])

    # ax2_3 = fig.add_axes([15.25/18, 1.5/18, 1.5/18, 4/18])
    ax2_3 = fig.add_axes([15.25/18, 1.5/8, 1.5/18, 6/8])

    night_mean_tt = []
    day_mean_tt = []
    for fish_nr in range(len(d_transition_times)):
        mean_tt = np.sum(day_d_transition_times[fish_nr]**2) / np.sum(day_d_transition_times[fish_nr])
        day_mean_tt.append(mean_tt)

        mean_tt = np.sum(night_d_transition_times[fish_nr]**2) / np.sum(night_d_transition_times[fish_nr])
        night_mean_tt.append(mean_tt)
    bp = ax2_3.boxplot([np.array(day_mean_tt[:6]) / 60., np.array(night_mean_tt[:6]) / 60., [], np.array(day_mean_tt[6:]) / 60., np.array(night_mean_tt[6:]) / 60.], sym='', patch_artist=True)
    for enu, patch in enumerate(bp['boxes']):
        if enu in [1, 4]:
            patch.set(facecolor='#888888')
        elif enu in [0, 3]:
            patch.set(facecolor='cornflowerblue')
        else:
            pass
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='k')

    ax2_3.set_xticks([1.5, 4.5])
    ax2_3.set_xticklabels(['male', 'female'], rotation = 45)
    ax2_3.set_ylim([0, 50])
    ax2_3.set_yticks(np.arange(0, 51, 10))
    ax2_3.set_yticklabels(np.arange(0, 51, 10))


    ax2_3.set_ylabel('weighted transition time [min]', fontsize=fs)

    # fig.text(13.75 / 18, 5.5 / 18, 'G', ha='center', va='center', fontsize=fs + 6)
    fig.text(13.75 / 18, 7.5 / 8, 'C', ha='center', va='center', fontsize=fs + 6)

    if True:
        # MALEFEMALE
        print('\n\n ### figure 2 wighted ####')
        # sens_cutoff = [4, 5, 6, 7, 8]
        sens_cutoff = [6]
        for sco in sens_cutoff:
            print('\n %.0f males // %.0f females' % (sco, 14 - sco))
            print('\n Mann whitney U transition rate day / night')
            stat, p = scp.mannwhitneyu(day_mean_tt[:sco], night_mean_tt[:sco])
            d = cohans_d(day_mean_tt[:sco], night_mean_tt[:sco])
            print('Male: U = %.3f, p = %.3f, d = %.2f' % (stat, p*2, d))
            stat, p = scp.mannwhitneyu(day_mean_tt[sco:], night_mean_tt[sco:])
            d = cohans_d(day_mean_tt[sco:], night_mean_tt[sco:])
            print('Female: U = %.3f, p = %.3f, d = %.2f' % (stat, p*2, d))

            print('\n Mann whitney U transition rate day / night')
            stat, p = scp.mannwhitneyu(day_mean_tt[:sco], day_mean_tt[sco:])
            d = cohans_d(day_mean_tt[:sco], day_mean_tt[sco:])
            print('Day: U = %.3f, p = %.3f, d = %.2f' % (stat, p*2, d))
            stat, p = scp.mannwhitneyu(night_mean_tt[sco:], night_mean_tt[:sco])
            d = cohans_d(night_mean_tt[sco:], night_mean_tt[:sco])
            print('Night: U = %.3f, p = %.3f, d = %.2f' % (stat, p*2, d))

            print('#########################')

    ax2_3.text(1.5, 45, '**', va = 'center', ha='center', fontsize=fs)
    ax2_3.text(4.5, 45, '*', va = 'center', ha='center', fontsize=fs)

    for ax in [ax2_0, ax2_0a, ax2_1a, ax2_1, ax2_2, ax2_3]:
        ax.tick_params(labelsize=fs - 1)

    for ax in [ax2_0, ax2_0a, ax2_1a, ax2_1, ax2_2, ax2_3]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    fig.savefig('/home/raab/paper_create/2019raab_habitats/transition_times.jpg', dpi=300)

    #########################################################################

    # ax2_1.set_xlabel('Fish ID')

    # ax2_0.set_xticks([])
    # ax2_0x.set_xlim([-0.5, 13.5])
    # ax2_0x.set_xticks([])
    # ax2_1.set_xlim([-0.5, 13.5])
    # ax2_1.set_xticks([])
    # ax2_1x.set_xlim([-0.5, 13.5])
    # ax2_1x.set_xticks(np.arange(14))

    # ax2_0.set_yticks(np.arange(0, 1.2, .2))
    # # ax2_1.set_yticks(np.arange(0.2, 1.2, .2))
    #
    # ax2_0x.set_yticks([5000, 10000])
    # ax2_0x.set_yticklabels(['5k', '10k'])
    # # ax2_1x.set_yticks([5000, 10000])
    # ax2_1x.set_yticklabels(['5k', '10k'])
    # ax2_1x.set_xlabel('Fish no.', fontsize=10)



    # ax2_1x.invert_yaxis()

    #### FIGURE 1 - END####
    embed()
    quit()


    # fig, ax = plt.subplots(facecolor='white', figsize=(20./2.54, 12/2.54))

    # n_fig, n_ax = plt.subplots(facecolor='white', figsize=(20./2.54, 12/2.54))
    day_n_fig, day_n_ax = plt.subplots(facecolor='white', figsize=(20./2.54, 12/2.54))
    day_n_ax.set_title('day occupation')

    night_n_fig, night_n_ax = plt.subplots(facecolor='white', figsize=(20./2.54, 12/2.54))
    night_n_ax.set_title('night occupation')

    for hab_nr in range(len(hab_fish_count)):
        # ax.plot(bin_t, np.nansum(binned_hab_fish_count[:hab_nr+1], axis = 0), color=hab_colors[hab_nr])
        # ax.plot(bin_t, binned_hab_fish_count[hab_nr], color=hab_colors[hab_nr])

        # n_ax.plot(bin_t, np.nansum(binned_hab_fish_count[:hab_nr+1], axis = 0) / np.nansum(binned_hab_fish_count, axis = 0), color=hab_colors[hab_nr])

        day_n_ax.plot(np.array(bin_t)[bin_t_day_mask], np.nansum(np.array(binned_hab_fish_count)[:hab_nr+1, bin_t_day_mask], axis = 0) / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_day_mask], axis = 0), '-', color=hab_colors[hab_nr])
        # day_n_ax.plot(np.array(bin_t)[bin_t_day_mask], np.array(binned_hab_fish_count)[hab_nr, bin_t_day_mask] / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_day_mask], axis = 0), '-', color=hab_colors[hab_nr], marker='.')
        night_n_ax.plot(np.array(bin_t)[bin_t_night_mask], np.nansum(np.array(binned_hab_fish_count)[:hab_nr+1, bin_t_night_mask], axis = 0) / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_night_mask], axis = 0), '-', color=hab_colors[hab_nr])
        # night_n_ax.plot(np.array(bin_t)[bin_t_night_mask], np.array(binned_hab_fish_count)[hab_nr, bin_t_night_mask] / np.nansum(np.array(binned_hab_fish_count)[:, bin_t_night_mask], axis = 0), '--', color=hab_colors[hab_nr], marker='.')

        # n_ax.plot(bin_t, np.nansum(binned_hab_fish_count[:hab_nr+1], axis = 0) / np.nansum(binned_hab_fish_count, axis = 0), color=hab_colors[hab_nr])
        # n_ax.plot(bin_t, np.nansum(binned_hab_fish_count[:hab_nr+1], axis = 0) / np.nansum(binned_hab_fish_count, axis = 0), color=hab_colors[hab_nr])
    # plt.show()

    # fish specific analysis

    fig = plt.figure(facecolor='white', figsize=(20/2.54, 14/2.54))

    ax0 = fig.add_axes([.1, .1, .395, .8])
    ax1 = fig.add_axes([.505, .1, .395, .8])
    ax0.set_yticks(np.arange(14))
    ax0.set_yticklabels(np.arange(14)+1)
    ax0.set_ylabel('# fish', fontsize=12)
    fig.text(.5, .025, 'habitat occupation likelihood', ha='center', va='center', fontsize=12)

    ax0.tick_params(labelsize=10)
    ax1.tick_params(labelsize=10)

    ax1.set_yticks([])
    ax0.set_xticks(np.arange(0.2, 1.2, .2))
    ax1.set_xticks(np.arange(0.2, 1.2, .2))
    ax0.invert_xaxis()

    ax1.fill_between([0, 1], [-1, -1], [14, 14], color='k', alpha = .6)

    for fish_nr in range(len(rel_d_fish_counts_in_habitat)):
        day_upshift = 0
        night_upshift = 0

        for hab_nr in range(len(rel_d_fish_counts_in_habitat[fish_nr])):
            ax0.barh(fish_nr, rel_d_fish_counts_in_habitat[fish_nr][hab_nr], left=day_upshift, color=hab_colors[hab_nr], height=0.5, edgecolor='k', lw=.6)
            day_upshift += rel_d_fish_counts_in_habitat[fish_nr][hab_nr]

            ax1.barh(fish_nr, rel_n_fish_counts_in_habitat[fish_nr][hab_nr], left=night_upshift, color=hab_colors[hab_nr], height=0.5, edgecolor='k', lw=.6)
            night_upshift += rel_n_fish_counts_in_habitat[fish_nr][hab_nr]

    ax0.invert_yaxis()
    ax1.invert_yaxis()

    ax0.set_xlim([1, 0])
    ax1.set_xlim([0, 1])

    # ax0.set_ylim([-1, 14])
    ax1.set_ylim([-.5, 13.5])
    ax0.set_ylim([-.5, 13.5])


    # plt.tight_layout()


    ##############################################################################################

    # fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    # for fish_nr in range(len(rel_d_fish_counts_in_habitat)):
    #     # embed()
    #     upshift = 0
    #     if np.sum(rel_d_fish_counts_in_habitat[fish_nr]) == 0:
    #         continue
    #     for hab_nr in range(len(rel_d_fish_counts_in_habitat[fish_nr])):
    #         ax.bar(len(rel_n_fish_counts_in_habitat) - fish_nr, rel_d_fish_counts_in_habitat[fish_nr][hab_nr], bottom=upshift, color=hab_colors[hab_nr])
    #         upshift += rel_d_fish_counts_in_habitat[fish_nr][hab_nr]
    #     ax.set_title('day occupation of habitats')
    #     ax.set_xlabel('fish Nr.')
    #     ax.set_ylabel('rel. occurance in habitat')
    #     # ax.set_ylim([0, 1])
    # plt.tight_layout()
    # # fig.savefig(saving_folder + 'total_occupation_in_habitats_day.pdf')
    # # plt.close()
    #
    # fig, ax = plt.subplots(facecolor='white', figsize=(20/2.54, 12/2.54))
    # for fish_nr in range(len(rel_n_fish_counts_in_habitat)):
    #     # embed()
    #     upshift = 0
    #     if np.sum(rel_n_fish_counts_in_habitat[fish_nr]) == 0:
    #         continue
    #     for hab_nr in range(len(rel_n_fish_counts_in_habitat[fish_nr])):
    #         ax.bar(fish_nr, rel_n_fish_counts_in_habitat[fish_nr][hab_nr], bottom=upshift, color=hab_colors[hab_nr])
    #         upshift += rel_n_fish_counts_in_habitat[fish_nr][hab_nr]
    #     ax.set_title('night occupation of habitats')
    #     ax.set_xlabel('fish Nr.')
    #     ax.set_ylabel('rel. occurance in habitat')
    #     # ax.set_ylim([0, 1])
    # plt.tight_layout()
    # # fig.savefig(saving_folder + 'total_occupation_in_habitats_night.pdf')
    # # plt.close()
    # plt.show()

if __name__ == '__main__':
    main()