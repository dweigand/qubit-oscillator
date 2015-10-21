import time                                                   # used for timing

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import scipy.integrate
import scipy.special
import scipy.misc
import numpy as np
import itertools
from qutip import *

import feedback as fb
import state_simulation as sim
import misc

def error_rate(n_m, mode):
    state, P = sim.final_state(result_list, psi0,
                               mode, method='weighed circuit')


if __name__ == '__main__':
    # unittest.main(verbosity=2)

    print("Simulating feedback with weighing on ideal code")
    Mmax = 9

    numBins = 500
    plotcut = 4
    plotList = [4, 5, 6, 7, 8]
    delta = 0.14
    r = np.log(1/delta)                      # computes the squeezing factor
    psi0 = tensor(squeeze(150, r) * basis(150, 0), basis(2, 0)).unit()
    # initialize
    plt.close()

    mode_list = ['rpe', 'arpe', 'off', 'random']
    delta_theta = np.zeros((Mmax, 2**Mmax))
    hist_array = np.zeros((4, Mmax, numBins))
    bin_array = np.zeros((4, Mmax, numBins+1))

    # compute
    for j, mode in enumerate(mode_list):
        print('mode is '+mode)
        P_array = [np.zeros(2**m) for m in np.arange(1, Mmax+1)]
        weight_array = [np.zeros(2**m) for m in np.arange(1, Mmax+1)]
        for i, m in enumerate(plotList):
            print('m is '+str(m))
            result_list = result_range(0, 2**m, m)
            state_list, w_list = list(zip(*serial_map(sim.final_state,
                                                      result_list,
                                                      task_args=(psi0, mode),
                                  task_kwargs={'method': 'weighed circuit'})))
            print('states obtained')
            # This wastes a lot of time, but is fast enough for now
            _, _, _, _, _, _, _, p_z, _, _ =\
            sim.analysis_wrapper(state_list, 20, 1025, np.sqrt(np.pi)/6)
            hist, bins = np.histogram(np.clip(1-p_z, 0, 0.2),
                                      numBins, range=(0, 1), density=True,
                                      weights=w_list)
            hist_array[j, i] = hist/numBins
            bin_array[j, i] = bins
        print(mode+' done')
    print('computing done')
    hist_array = hist_array.transpose(1, 0, 2)
    bin_array = bin_array.transpose(1, 0, 2)

    # initialize plotting
    fig, ax_array = plt.subplots(len(plotList), 4, sharex='col', sharey='row')
    fig.set_size_inches(8.75*2, 11)
    plt.rc('font', size=14, **{'family': 'sans-serif',
                               'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    ylabels = [r"${0}$".format(0.1*x) for x in np.arange(11)]
    xlabels = [r"${0}$".format(0.05*x) for x in np.arange(5)]
    xlabels[-1] = r'$> 0.2$'

    # plot
    for j, _ in enumerate(mode_list):
        for i, m in enumerate(plotList):
            bins = bin_array[i][mode]
            hist = hist_array[i][mode]
            widths = np.diff(bins)
            widths[100] = 0.03
            ax = ax_array[i][mode]
            ax.bar(bins[:-1], hist, widths, color="None")
            ax.set_xlim([0, 0.21])
            plt.xticks(0.05 * np.arange(5))
            plt.yticks(0.1 * np.arange(11))
            ax.set_xticklabels(xlabels)
            ax.set_yticklabels(ylabels)
            ax.set_ylim([0, 0.6])
            if mode == 0:
                ax.set_ylabel(r'$P$', rotation='horizontal')
                ax.yaxis.labelpad = 20
            if mode == 3:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(str(m), fontsize=16, rotation='horizontal')
                ax.yaxis.labelpad = 31
            if m == max(plotList):
                ax.set_xlabel(r'$P_{\mathrm{error,p}}^{\sqrt{\pi}/6}$')
    ax_array[0][0].set_title(r'RPE', fontsize=20)
    ax_array[0][1].set_title(r'ARPE', fontsize=20)
    ax_array[0][2].set_title(r'OFF', fontsize=20)
    ax_array[0][3].set_title(r'RAND', fontsize=20)
    plt.suptitle(r'M', fontsize=20, x=0.95, y=0.92)
    # fig.savefig('histograms2.eps')
    plt.show()
