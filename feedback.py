# -*- coding: utf-8 -*-

"""
This module provides functions for several types of feedback

mode 0: Kitaev phase estimation. arxiv: quant-ph/9511026
mode 1: Adaptive repeated phase estimation PRA 63, 053804 (2001)
mode 2: Feedback off, Returns 0.
mode 3: returns a random number depending on m, x

More information can be found in arxiv: 1506.05033 (our paper) or my
M.Sc. thesis

Algorithm for Arpe:
    Following PRA 63, 053804 (2001) (readable summeries in our paper or my
    thesis), we optimize the Holevo variance of P(theta|n_m) where theta is
    the eigenvalue estimate of the measurement given the bit-string
    of results n_m.

    The probability distribution given result n_m is computed in P(m, x, mode)
    The function _S_optimize(phi, m, x, mode) computes the Holevo variance of
    and then averages over P(m+1, x, mode) and P(m+1, x x + 2^(m+1), mode),
    (in this example x < 2^m).
    Then, the optimal feedback is found in feedback_function(m, x, mode=1)
    using scipy.optimize.
    The function feedback(m, x, mode) is a wrapper to save time, as the
    feedback depends only on previous results, but x is the current result
    (thus holds e.g. feedback(1,0,mode)==feedback(1,1,mode) and so on)

    The wave function in u, v basis after the protocol can be obtained
    analytically and is implemented in psiv_noc(v, m, x, mode).
    The optimal estimate for the eigenvalue can be obtained by an integral over
    P(m, x, mode), this is done in estimate(m, x, mode)
    Using this estimate, the center of the wave function is shifted to v=0 in
    psiv(v, m, x, mode).

    With the corrected wave function, the probability to obtain a good codeword
    is obtained in P_no_error(m, x, mode), which integrates over psiv() in the
    interval v \in (-\sqrt{\pi}/6,\sqrt{\pi}/6).

    If the module is called as script, P_no_error() is evaluated for all 0<m<9,
    all x and all modes, the results are shown as a histogram.

Conventions:
    n_m: List of the binary measurement results. Entry m is the result of the
            m-th round.
    m: All functions take a round m as argument. This truncates the bit-string
        n_m, to the first m entries.
        e.g. P(m=1,x=1,mode) = P(m=1, x=7, mode)
    x: Some functions are memoized. For efficient memoizing, a function cannot
        take lists as input. Thus, the list n_m is converted to the integer x
"""

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt4Agg')

from scipy.special import binom
import numpy as np
import scipy.integrate as int
import scipy.optimize as opt
import time
import matplotlib.pylab as plt
import sys
import itertools as it

import unittest

from misc import Memoize
from qutip import *

import test_data


# Create a list of random results when this module is imported.
# We want the results to be random, but consistent through multiple calls
# with the same arguments.
random_list = np.random.random_sample((10, 2**10))*2*np.pi


M = 10
theta = np.linspace(0, 2*np.pi, 2**M+1)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TEST


class TestFeedback(unittest.TestCase):
    def test_feedback(self):
        for m in np.arange(1, 5):
            for x in np.arange(2**(m-1)):
                self.assertEqual(feedback(m, 2*x, 3), feedback(m, 2*x+1, 3))

    def test_feedback_function_rpe(self):
        self.assertEqual(feedback_function(2, 1, 0), 0)
        self.assertEqual(feedback_function(3, 1, 0), np.pi/2)
        for m in np.arange(1, 5):
            for x in np.arange(2**m):
                self.assertEqual(feedback_function(m, x, 0),
                                 np.mod(m*np.pi/2, np.pi))

    def test_feedback_function_arpe(self):
        self.assertEqual(feedback_function(0, 0, 1), 0)
        self.assertAlmostEqual(
            feedback_function(1, 1, 1),
            feedback_function(1, 0, 1),
            places=5
            )

        self.assertAlmostEqual(
            feedback_function(1, 0, 1),
            4.712388950181709,
            places=5
            )
        for m in np.arange(1, 5):
            for x in np.arange(2**m):
                self.assertAlmostEqual(
                    feedback_function(m, x, 1),
                    test_data.feedback_data[m-1][x],
                    places=5
                    )

    def test_feedback_function_off(self):
        for m in np.arange(1, 5):
            for x in np.arange(2**m):
                self.assertEqual(feedback(m, x, 2), 0)

    def test_feedback_function_random(self):
        for m in np.arange(5):
            for x in np.arange(2**m):
                self.assertEqual(feedback_function(m, x, 3), random_list[m, x])

    def test__distrib(self):
        for x in np.arange(2**4):
            for j in np.arange(5):
                self.assertAlmostEqual(
                    np.abs(_distrib(4, x, j, 1)),
                    np.abs(test_data.distrib_data[x, j]),
                    places=5)

# class TestStringMethods(unittest.TestCase):

# def test_upper(self):
    # self.assertEqual('foo'.upper(), 'FOO')

# def test_isupper(self):
    # self.assertTrue('FOO'.isupper())
    # self.assertFalse('Foo'.isupper())

# def test_split(self):
    # s = 'hello world'
    # self.assertEqual(s.split(), ['hello', 'world'])
    # check that s.split fails when the separator is not a string
    # with self.assertRaises(TypeError):
        # s.split(2)

# if __name__ == '__main__':
    # unittest.main()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# FUNCTIONS

# Checked, OK
@Memoize
def P(m, x, mode):
    """
    Compute the probability P(theta|x) of initial phase theta given results x.

    Args:
        m: Integer, current round of the protocol
        x: Integer, measurement results of the whole protocol
        mode: Integer, mode for feedback

    Returns:
        Array P(theta|x)
    """
    if m == 0:
        return 1
    p = P(m-1, np.floor_divide(x, 2), mode) * np.power(
            np.cos(0.5*(np.pi*np.mod(x, 2) + feedback(m, x, mode)+theta)), 2)
    return p


# phase estimate of result x at round m
# Checked
@Memoize
def estimate(m, x, mode):
    """
    Compute the estimate of the protocol given results x.

    Uses the algorithm of Berry, Wiseman, Breslin, PRA 63, 053804 (2001).
    More information can be found in arxiv: 1506.05033 (our paper) or my
    M.Sc. thesis

    Args:
        m: Integer, current round of the protocol
        x: Integer, measurement results of the whole protocol
        mode: Integer, mode for feedback

    Returns:
        Float, best estimate given the results
    """

    # Note that the numpy doc on np.angle is wrong,
    # the angle is in the interval (-pi,pi).

    p = P(m, x, mode)*np.exp(1j*theta)
    return np.angle(int.romb(p, 2*np.pi/(2**M + 1)))


# -----------------------------------------------------------------------------
# FEEDBACK

# Functions needed to obtain the feedback in various protocols are defined here
# An explanation of the protocols is found in the doc of feedback_function

def feedback(m, x, mode):
    """
    wrapper for feedback_function, saves some time (f(1,0) = f(1,1) etc.)

    Args:
        m: Integer, current round of the protocol
        x: Integer, measurement results of the whole protocol
        mode: Integer, mode for feedback

    Returns:
        Float, feedback
    """
    if m == 0:
        print("m must not be 0")
        exit()
    return feedback_function(m-1, np.floor_divide(x, 2), mode)


# Checked, seems good
@Memoize
def feedback_function(m, x, mode):
    """
    Compute feedback depending on mode (rpe, arpe, off). wrapped by feedback()

    mode 0: Kitaev phase estimation. arxiv: quant-ph/9511026
    mode 1: Adaptive repeated phase estimation PRA 63, 053804 (2001)
    mode 2: Feedback off, Returns 0.
    mode 3: returns a random number depending on m, x

    More information can be found in arxiv: 1506.05033 (our paper) or my
    M.Sc. thesis

    Args:
        m: Integer, current round of the protocol
        x: Integer, measurement results of the whole protocol
        mode: Integer, mode for feedback

    Returns:
        Float, feedback
    """
    if mode == 0:
        return np.mod(m*np.pi/2, np.pi)

    elif mode == 1:
        if m == 0:
            # if no measurement has been made, any feedback is ok. Choose 0.
            return 0
        return opt.fminbound(_S_optimize, 0, 2*np.pi, args=(m, x, mode))

    elif mode == 2:
        return 0

    elif mode == 3:
        return random_list[m, x]


def _S_optimize(phi, m, x, mode):
    """
    Form Average of the Holevo variance over results u=0,1

    Helper function for adaptive repeated phase estimation.
    More information on the algorithm can be found in arxiv/1506.05033 (our
    paper), or Berry, Wiseman, Breslin, PRA 63, 053804 (2001).

    Args:
        phi: float, 'true' phase that is measured
        m: Integer, current round of the protocol
        x: Integer, measurement results of the whole protocol
        mode: Integer, mode for feedback

    Returns:
        Float, feedback
    """
    S_sum = np.float(0)
    for u in np.arange(2):
        S = np.exp(1j*theta)*P(m, x, mode)*np.power(
                                      np.cos(0.5*(np.pi*u+phi+theta)), 2)
        S_sum += np.abs(int.romb(S, 2*np.pi/(2**M + 1)))
    return -S_sum

# -----------------------------------------------------------------------------
# STATE


def psiv(v, m, x, mode):
    """
    Return the corrected state after the idealized protocol u, v basis.

    Return the state in u,v basis after applying a correcting shift obtained
    by estimate().
    The function assumes a delta-peak as initial state (u=0). This can be done
    as the errors in u and v commute and u is computed with other means.
    This assumption is tested and holds.

    Args:
        v: Linspace in interval (-np.sqrt(np.pi)/6, np.sqrt(np.pi)/6),
            shift error v.
        m: Integer, current round of the protocol
        x: Integer, measurement results of the whole protocol
        mode: Integer, mode for feedback

    Returns:
        Array, Abs**2 of the wavefunction for the values of v
    """

    # The estimate is added, as we measure -v, where v is the shift (see paper)
    vc = v + estimate(m, x, mode)/(2*np.sqrt(np.pi))
    return psiv_noc(vc, m, x, mode)


def psiv_noc(v, m, x, mode):
    """
    Return the uncorrected state after the idealized protocol u, v basis.

    Return the state in u,v basis.
    The function assumes a delta-peak as initial state (u=0). This can be done
    as the errors in u and v commute and u is computed with other means.
    This assumption is tested and holds.

    Args:
        v: Linspace in interval (-np.sqrt(np.pi)/6, np.sqrt(np.pi)/6),
            shift error v.
        m: Integer, current round of the protocol
        x: Integer, measurement results of the whole protocol
        mode: Integer, mode for feedback

    Returns:
        Array, Abs**2 of the wavefunction for the values of v
    """
    psi = np.zeros(len(v), dtype=complex)
    for s in np.arange(m+1):
        psi += np.exp(1j*np.sqrt(np.pi)*2*s*(v))*_distrib(m, x, s, mode)
    return psi/_norm(m, x, mode)


# Checked, seems good
# checked second time, in depth. algorithm is OK.
@Memoize
def _distrib(m, x, j, mode):
    """
    Obtain distribution of squeezed states forming a code state defined by x

    More information can be found in arxiv: 1506.05033 (our paper) or my
    M.Sc. thesis

    Args:
        m: Integer, current round of the protocol
        x: Integer, measurement results of the whole protocol
        j: Integer, running index of the squeezed state
        mode: Mode (rpe, arpe, off) for feedback integer 0,1 or 2

    Returns:
        Complex, Weight of the squeezed state j in a code state
    """
    if j == 0:
        return 1
    n_comb = np.int(binom(m, j))
    phases = np.zeros(m, dtype=complex)
    for i in np.arange(1, m+1):
        x_current = np.floor_divide(x, 2**(m-i))
        phases[i-1] = feedback(i, x_current, mode) + np.mod(x_current, 2)*np.pi
    t = it.chain.from_iterable(it.combinations(phases, j))
    t = np.fromiter(t, dtype=complex, count=n_comb*j)
    t.resize((n_comb, j))
    t = np.exp(1j*t.sum(axis=1))
    return t.sum()


def _norm(m, x, mode):
    """
    Approximate the norm of a code state after m measurements with result x

    The function assumes that the squeezed states are approximately orthogonal.
    If a codeword is even slightly useful, this assumption is good to 10^-9.

    Args:
        m: Integer, current round of the protocol
        x: Integer, measurement results of the whole protocol
        mode: Mode (rpe, arpe, off) for feedback integer 0,1 or 2

    Returns:
        Float, norm of the state
    """
    nrm = 0
    for j in np.arange(m+1):
        nrm += np.power(np.abs(_distrib(m, x, j, mode)), 2)
    return np.sqrt(nrm*np.sqrt(np.pi))

# -----------------------------------------------------------------------------
# ANALYSIS


@Memoize
def P_no_error(m, x, mode):
    """
    Compute the probability to obtain a good code state given m, x and mode

    Args:
        m: Integer, current round of the protocol
        x: Integer, measurement results of the whole protocol
        mode: Mode (rpe, arpe, off) for feedback integer 0,1 or 2

    Returns:
        Float, probability
    """
    v = np.linspace(-np.sqrt(np.pi)/6, np.sqrt(np.pi)/6, 2**M + 1)
    psi = np.power(np.abs(psiv(v, m, x, mode)), 2)
    return int.romb(psi, np.sqrt(np.pi)/(3*(2**M + 1)))


# -----------------------------------------------------------------------------
# MISC

def _parallel_helper(x, m, mode):
    feedback_function(m, x, mode)
    estimate(m, x, mode)
    return 1 - P_no_error(m, x, mode)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# COMPUTE HISTOGRAM

if __name__ == '__main__':
    # unittest.main(verbosity=2)

    print("Simulating feedback on ideal code")
    Mmax = 9

    numBins = 500
    plotcut = 4
    plotList = [4, 5, 6, 7, 8]

    # initialize
    plt.close()

    delta_theta = np.zeros((Mmax, 2**Mmax))
    hist_array = np.zeros((4, Mmax, numBins))
    bin_array = np.zeros((4, Mmax, numBins+1))
    plotListmax = max(plotList)

    # compute
    for mode in np.arange(4):
        mode_name = ['rpe', 'arpe', 'off', 'random'][mode]
        P_array = [np.zeros(2**m) for m in np.arange(Mmax)]
        for m in np.arange(Mmax):
            # print(m)
            x = np.arange(2**m)
            P_array[m] = parallel_map(_parallel_helper, x,
                                      task_args=(m, mode))
            # print(np.mean(P_array[m]))
            hist, bins = np.histogram(np.clip(P_array[m], 0, 0.2),
                                      numBins, range=(0, 1), density=True)
            hist_array[mode, m] = hist/numBins
            bin_array[mode, m] = bins
        print(mode_name+' done')
    print('computing done')
    hist_array = hist_array.transpose(1, 0, 2)
    bin_array = bin_array.transpose(1, 0, 2)

    # initialize plotting
    fig, ax_array = plt.subplots(len(plotList), 4, sharex='col', sharey='row')
    fig.set_size_inches(8.75*2, 11)
    plt.rc('font', size=14, **{'family': 'sans-serif',
                               'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    ylabels = [r"${0}$".format(0.1*x) for x in range(11)]
    xlabels = [r"${0}$".format(0.05*x) for x in range(5)]
    xlabels[-1] = r'$> 0.2$'

    # plot
    for mode in np.arange(4):
        for i, m in enumerate(plotList):
            bins = bin_array[m][mode]
            hist = hist_array[m][mode]
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
            if m == plotListmax:
                ax.set_xlabel(r'$P_{\mathrm{error,p}}^{\sqrt{\pi}/6}$')
    ax_array[0][0].set_title(r'RPE', fontsize=20)
    ax_array[0][1].set_title(r'ARPE', fontsize=20)
    ax_array[0][2].set_title(r'OFF', fontsize=20)
    ax_array[0][3].set_title(r'RAND', fontsize=20)
    plt.suptitle(r'M', fontsize=20, x=0.95, y=0.92)
    # fig.savefig('histograms2.eps')
    plt.show()
