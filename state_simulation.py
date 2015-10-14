# -*- coding: utf-8 -*-

"""
Provides an explicit simulation of the protocol given feedback mode, fixed n_m

mode 0: Kitaev phase estimation. arxiv: quant-ph/9511026
mode 1: Adaptive repeated phase estimation PRA 63, 053804 (2001)
mode 2: Feedback off, Returns 0.
mode 3: returns a random number depending on m, x

More information can be found in arxiv: 1506.05033 (our paper) or my
M.Sc. thesis

The module simulates the protocol using either a perfect circuit
perfect_protocol() or by using Hamiltonian time evolution protocol().

The feedback is provided by feedback.py

Different to protocol.pu, the module provides the simulation and in-depth
analysis of few fixed measurement outcomes at a time.

At the moment, swapping is done by calling one function or the other, they use
the same arguments.

Conventions:
    n_m: List of the binary measurement results. Entry m is the result of the
            m-th round.
    m: All functions take a round m as argument. This truncates the bit-string
        n_m, to the first m entries.
        e.g. P(m=1,x=1,mode) = P(m=1, x=7, mode)
    x: Some functions are memoized. For efficient memoizing, a function cannot
        take lists as input. Thus, the list n_m is converted to the integer x
"""

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

import feedback as fb
from qutip import *


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# SETTINGS

N = 150                                               # dimension of Fock space
n_points_plot = 1025
error_threshold = np.sqrt(np.pi)/6


# -----------------------------------------------------------------------------
# SETTINGS FOR TIME-EVOLUTION

# Experimental properties.                          All frequencies in GHz
wc = 8.2 * 2 * np.pi                                # bare freq of the cavity
wq = 7.46 * 2 * np.pi                               # bare freq of the qubit
# g = 0.01 * 2 * np.pi                              # can be used to set g
# chi = g**2/(np.abs(wc-wq))                        # chi if g is set
chi = 2.4/1000 * 2 * np.pi                          # chi
K_s = 3.61/1000000 * 2 * np.pi                      # Kerr
K_qs = 4.2/1000000 * 2 * np.pi                      # Qubit-dependent Kerr

# other
tlist = np.linspace(0, np.pi/(2*chi), 513)    # List of times can be plotted

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# QUBIT-CAVITY OPERATORS


I = tensor(qeye(N), qeye(2))                         # Identity Operator

# CAVITY operators
a = tensor(destroy(N), qeye(2))                      # anihilation operator
nc = a.dag() * a                                     # number operator
nnc = a.dag() * a.dag() * a * a                      # Kerr operator


def displace_(alpha):                               # Displacement operator
    return tensor(displace(N, alpha), qeye(2))

# QUBIT operators
b = tensor(qeye(N), destroy(2))                     # anihilation operator
nq = b.dag() * b                                    # number operator
sz = tensor(qeye(N), sigmaz())                      # Pauli Z
sx = tensor(qeye(N), sigmax())                      # Pauli X
hadamard = tensor(qeye(N), snot())                  # Hadamard Gate


def phasegate_(phi):                                # Phasegate
    return tensor(qeye(N), phasegate(phi))


# Projector onto qubit state 0/1. Changes normalization !
def projector(x):
    return tensor(qeye(N), fock_dm(2, x))


# QUBIT-CAVITY operators


# Controlled rotation /w Time Evolution
def controlled_rotation(state, H):
    """
    Controlled rotation based on time evolution using the Hamiltonian H

    Args:
        state: state to be rotated
        H: Hamiltonian used in time-evolution
    Returns:
        Qutip result object, result.states[-1] is the state after evolution
    """
    # Placeholder to show how time-dependent Hamiltonians work
    # H_t = [[Hc, wc_t], [Hq, wq_t], -chi * Hqc]
    result = mesolve(H, state, tlist, [], [],
                     options=Odeoptions(nsteps=5000))
    return result


# Controlled rotation (instant)
def controlled_zgate(U):
    return tensor(U, fock_dm(2, 1)) + tensor(U.dag(), fock_dm(2, 0))


def qubit_reset(state, x):
    """
    Project the qubit to state x and then reset it to 0

    Acts on the joint qubit-cavity state 'state'.

    Args:
        state: Qutip state object, joint cavity-qubit state.
        x: Integer 0 or 1, measurement result

    Returns:
        Qutip state
    """
    state = projector(x) * state
    if x == 0:
        return state.unit()

    elif x == 1:
        state = sx * state
        return state.unit()


def remove_qubit(state):
    """
    Remove the qubit from the joint system. Assumes qubit_reset() has been run

    Args:
        state: Qutip state object, joint cavity-qubit state, must be pure

    Returns:
        Qutip state
    """
    data = state.data[::2]
    state = Qobj(data)
    return state.unit()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Protocol

# The evolution is done according to Fig 10 in arxiv/1506.05033v2 (our paper)

def final_state(n_m, H, psi0, mode):
    """
    Return the state after a series of measurements with result-string n_m

    Args:
        n_m: Binary List, bit-string of results. n_m[m] is the m-th round.
        H: Hamiltonian used for time evolution
        psi0: Qutip state, initial state
        mode: mode used for feedback, see feedback.feedback()

    Returns:
        Qutip state
    """
    psi = psi0
    for m in np.arange(1,len(n_m)+1):
        psi = protocol(psi, m, n_m, mode, H)
    psi = correcting_shift(m, n_m, mode, psi)
    psi = remove_qubit(psi)
    return psi

def protocol(state, m, n_m, mode, H):
    """
    Do the protocol in arxiv/1506.05033, Fig 5 on a state using time-evolution

    The circuit for the controlled displacement using time-evolution can be
    found in the same paper, Fig 10

    Args:
        state: Qutip state, initial state
        H: Hamiltonian used for time evolution
        n_m: Binary List, bit-string of results. n_m[m] is the m-th round.
        m: current round, starting at 0
        mode: mode used for feedback, see feedback.feedback()

    Returns:
        Qutip state
    """
    phi = fb.feedback(m, n_m, mode)
    state = hadamard * state
    result1 = controlled_rotation(state, H)
    state = result1.states[-1].unit()
    state = displace_(-1j * np.sqrt(np.pi/2)) * sx * state
    result2 = controlled_rotation(state, H)
    state = result2.states[-1].unit()
    state = hadamard * phasegate_(phi) * sx * state
    state = qubit_reset(state, n_m[m-1])
    return state.unit()


def perfect_protocol(state, m, n_m, mode, H):
    """
    Do the protocol in arxiv/1506.05033, Fig 5 on a state using the circuit

    Args:
        state: Qutip state, initial state
        H: Does nothing here, makes it easier to swap between the two protocols
        n_m: Binary List, bit-string of results. n_m[m] is the m-th round.
        m: current round, starting at 0
        mode: mode used for feedback, see feedback.feedback()

    Returns:
        Qutip state
    """
    distr = [(np.abs(fb._distrib(m, n_m, j, mode)),np.angle(fb._distrib(m, n_m, j, mode))) for j in np.arange(m+1)]
    phi = fb.feedback(m, n_m, mode)
    state = hadamard * state
    state = controlled_zgate(displace(N, np.sqrt(np.pi/2))) * state
    state = hadamard * phasegate_(phi) * state
    state = qubit_reset(state, n_m[m-1])
    return state.unit()


def correcting_shift(m, n_m, mode, state):
    """
    Correct the state based on the estimate for n_m.

    Currently not working correctly, as the correlation between n_m and state
    is not as expected

    Args:
        n_m: list of measurement results
        state: state obtained with measurement results n_m
        mode: mode used for feedback
    Returns:
        array of length x_inter
    """
    est = fb.estimate(m, n_m, mode)
    return (displace_(1j*est/(2*np.sqrt(2*np.pi))) * state).unit()


#def correcting_cheat(n_m, y_vec, pfunc, state):
    #"""
    #Correct the state based on an error analysis of pfunc

    #Deprecated, should not be used. Uses lots of computational power and is not
    #properly tested.

    #Args:
        #n_m: list of measurement results
        #y_vec: x-vector for pfunc
        #pfunc: wigner function in p-space
        #state: state to be corrected
    #Returns:
        #array of length x_inter
    #"""
    #y_vec = (np.mod(y_vec+np.sqrt(np.pi)/2,
                    #np.sqrt(np.pi))-np.sqrt(np.pi)/2)*2*np.sqrt(np.pi)
    #p = pfunc*np.exp(1j*y_vec)
    #est = np.angle(scipy.integrate.simps(p, y_vec))
    #print(n_m)
    #print(est/(2*np.sqrt(np.pi)))
    #return (displace_(-0) * state).unit()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ANALYSIS


def analysis_wrapper(states, lim, n_points_plot, threshold):
    """
    Return interesting functions obtained from the Wigner function of a state.

    Args:
        states: list of Qutip state objects
        lim: float, the vector x given to qutip.wigner() is in (-lim,lim)
        n_points_plot: Integer, Number of Points used in the vectors x and y.
        threshold: Threshold used to compute the error rate

    Returns:
        x_vec: np array, linspace of points in x
        y_vec: np array, linspace of points in y
        W: Wigner function at points x,y
        qfunc: Integral of W over p
        pfunc: Integral of W over q
        pfunc_m: pfunc, masked to (-threshold,threshold) mod sqrt(pi)
        qfunc_m: qfunc, masked to (-threshold,threshold) mod sqrt(pi)
        p_z: rate of v>threshold (z-type) errors
        p_x: rate of u>threshold (z-type) errors
        error_rate: combined rate
    """
    x_vec = np.linspace(-lim, lim, n_points_plot)
    W, yvecs = list(zip(*parallel_map(wigner, states,
                                    task_args=(x_vec, x_vec),
                                    task_kwargs={'method': 'fft'})))

    xvecs = np.repeat([x_vec, x_vec], [1, len(states)-1], axis=0)

    y_vec = yvecs[-1]
    qfunc, pfunc, qfunc_m, pfunc_m, p_z, p_x, error_rate = list(zip(
        *parallel_map(analysis_function,
                    list(zip(*[W, xvecs, yvecs])),
                    task_args=(threshold,))))

    return_list = map(np.asarray, (x_vec, y_vec, W,
                                          qfunc, pfunc, qfunc_m, pfunc_m,
                                          p_z, p_x, error_rate))
    return return_list


def analysis_function(W_tuple, threshold):
    """
    Return interesting functions obtained from the Wigner function of a state.

    Args:
        W_tuple: tuple of W, x_vec, y_vec as defined in analysis_wrapper
        threshold: Threshold used to compute the error rate

    Returns:
        qfunc: Integral of W over p
        pfunc: Integral of W over q
        pfunc_m: pfunc, masked to (-threshold,threshold) mod sqrt(pi)
        qfunc_m: qfunc, masked to (-threshold,threshold) mod sqrt(pi)
        p_z: rate of v>threshold (z-type) errors
        p_x: rate of u>threshold (z-type) errors
        error_rate: combined rate
    """
    w = W_tuple[0]
    x_vec = W_tuple[1]
    y_vec = W_tuple[2]
    y_dist = np.diff(y_vec)

    dx = np.mean([np.diff(x_vec).max(), np.diff(x_vec).min()])
    dy = np.mean([y_dist.max(), y_dist.min()])

    condlist_y = np.abs(np.mod(y_vec + np.sqrt(np.pi)/2,
                               np.sqrt(np.pi)) - np.sqrt(np.pi)/2) < threshold
    condlist_x = np.abs(np.mod(x_vec + np.sqrt(np.pi)/2,
                               np.sqrt(np.pi)) - np.sqrt(np.pi)/2) < threshold

    pfunc = scipy.integrate.romb(w, dx=dx, axis=1)
    qfunc = scipy.integrate.romb(w, dx=dy, axis=0)

    error = 1-scipy.integrate.romb(pfunc, dx=dy)
    if error > 0.001:
        print("rounding error of Wigner function is large: %6f" % error)

    pfunc_m = mask(pfunc, condlist_y)
    qfunc_m = mask(qfunc, condlist_x)
    p_z = scipy.integrate.romb(pfunc_m, dx=dy)
    p_x = scipy.integrate.romb(qfunc_m, dx=dx)
    error_rate = 1-p_x * p_z
    return qfunc, pfunc, qfunc_m, pfunc_m, p_z, p_x, error_rate


def wavefunc(state, x_vec):
    """
    Return the wavefunctions of a state in q and p using Hermite polynomials

    Args:
        state: some pure qutip state
        x_vec: array of x-values on which the wavefunctions are evaluated

    Returns:
        q: Array, wavefunction in q
        p: Array, wavefunction in p
    """
    N = state.dims[0][-1]
    q = np.zeros_like(x_vec, dtype=np.complex_)
    p = np.zeros_like(x_vec, dtype=np.complex_)
    cx = state.data.tocoo()
    for n,_,data in itertools.izip(cx.row, cx.col, cx.data):
        q += (data*np.exp(-x_vec**2/2.0)*scipy.special.eval_hermite(n,x_vec)
              /np.sqrt(scipy.misc.factorial(n)*2**n))
        p += (data*np.exp(-0.5*(x_vec**2+1j*np.pi*n))*scipy.special.eval_hermite(n,x_vec)
              /np.sqrt(scipy.misc.factorial(n)*2**n))
    return q, p

# -----------------------------------------------------------------------------
# Helper functions

def absmax(array):
    return abs(array).max()


def mask(choicelist, condlist, default=0):
    """
    Return an array of the same length as choicelist masked by condlist

    bit of a hack

    Args:
        choicelist: Array
        condlist: Boolean Array with same length as choicelist
        default: all entries j of choicelist with condlist[j] == False
            are set to default

    Returns:
        array
    """
    choicelist_list = [choicelist, choicelist]
    condlist_list = [condlist, condlist]
    return np.select(condlist_list, choicelist_list, default)


def result_range(Min, Max, m):
    """Return range min-max as bit strings of length m"""
    return np.arange(Min, Max)[:, np.newaxis] >> np.arange(m)[::-1] & 1


def result_random(N_results, m):
    """Return N_results random bit-strings of length m"""
    return np.random.random_integers(0,1,size=(N_results,m))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# SIMULATION
if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # SETTINGS
    method = 'histogram'
    start = time.time()
    print("start")
    delta = 0.14                             # initial squeezing
    mode = 1
    MMax = 8
    numBins = 500
    
    r = np.log(1/delta)                      # computes the squeezing factor
    lim = 20                                 # fixed for equal precision in p,q

    # Hamiltonians
    Hc = nc                                     # cavity evolution
    H2c = nnc                                   # cavity Kerr
    Hq = nq                                     # qubit evolution
    Hqc = nc * sz                               # qubit-cavity
    H2qc = nnc * sz
    H = - chi * Hqc  # - K_s * H2c - K_qs * H2qc # Hamiltonian for protocol

    result_list = result_range(0, 2**MMax, MMax)
    
    #print("result is %s" % str(n_m))
    if mode == 0:
        print("feedback using rpe")
    elif mode == 1:
        print("feedback using arpe")
    elif mode == 2:
        print("feedback off")
    
    # -------------------------------------------------------------------------
    # EVOLUTION
    
    psi0 = tensor(squeeze(N, r) * basis(N, 0), basis(2, 0)).unit()
    state_list = parallel_map(final_state, result_list,
                            task_args=(H, psi0, mode))
    print("obtained new state, time: %6f" % float(time.time()-start))
    
    # -------------------------------------------------------------------------
    # ANALYSIS
    
    x_vec, y_vec, W, qfunc, pfunc, qfunc_m, pfunc_m, p_z, p_x, error_rate =\
        analysis_wrapper(state_list, lim, n_points_plot, error_threshold)
    print("error analysis done, time: %6f" % float(time.time()-start))
    
    print("p_z")
    print(p_z)
    print("p_x")
    print(p_x)
    print("error rate")
    print(error_rate)
    
    # -------------------------------------------------------------------------
    # PLOTTING
    if method == 'histogram':
        hist, bins = np.histogram(np.clip(error_rate, 0, 0.2),
                                      numBins, range=(0, 1), density=True)
        hist = hist/numBins
        widths = np.diff(bins)

        fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
        
        fig.set_size_inches(8.75*2, 11)
        plt.rc('font', size=14, **{'family': 'sans-serif',
                                'sans-serif': ['Helvetica']})
        plt.rc('text', usetex=True)
        ylabels = [r"${0}$".format(0.1*x) for x in range(11)]
        xlabels = [r"${0}$".format(0.05*x) for x in range(5)]
        xlabels[-1] = r'$> 0.2$'
        widths[100] = 0.03
        ax.bar(bins[:-1], hist, widths, color="None")
        ax.set_xlim([0, 0.21])
        plt.xticks(0.05 * np.arange(5))
        plt.yticks(0.1 * np.arange(11))
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        ax.set_ylim([0, 0.6])
        ax.set_ylabel(r'$P$', rotation='horizontal')
        ax.yaxis.labelpad = 20
        ax.set_xlabel(r'$P_{\mathrm{error,p}}^{\sqrt{\pi}/6}$')
        print("plotting done, time: %6f" % float(time.time()-start))
        plt.show()
        exit()
    
    if method == 'single':

        pfunc_wrap = [misc.wrap_function(y_vec, pfunc[i], -inter_vec)
                for i in np.arange(len(pfunc))]
        
        func1 = qfunc
        vec1 = x_vec
        xlim1 = np.sqrt(np.pi)*6

        func2 = pfunc_wrap
        vec2 = inter_vec
        xlim2 = np.sqrt(np.pi)/2

        func3 = psi_c
        vec3 = inter_vec
        xlim3 = np.sqrt(np.pi)/2

        wlim = parallel_map(sim.absmax, W)
        ticks = np.arange(-10*np.ceil(lim/np.sqrt(np.pi)),
                            10*np.ceil(lim/np.sqrt(np.pi)) + 1, 1) * np.sqrt(np.pi)/2
        labels = ["" for i in ticks]

        global axes
        global fig
        fig, axes = plt.subplots(5, len(state_list), figsize=(12, 2))

        for i in np.arange(len(state_list)):
            axes[0][i].contourf(x_vec, y_vec, W[i], 100,
                                norm=matplotlib.colors.Normalize(-wlim[i],
                                                                    wlim[i]),
                                cmap=plt.get_cmap('RdBu'))
            axes[0][i].set_xlabel(r'Re $\alpha$', fontsize=18)
            axes[0][i].set_ylabel(r'Im $\alpha$', fontsize=18)
            # axes[0][i].set_xticks(ticks* np.sqrt(np.pi))
            # axes[0][i].set_xticklabels(labels, fontsize=20)
            # axes[0][i].set_yticks(ticks * np.sqrt(np.pi))
            # axes[0][i].set_yticklabels(labels, fontsize=20)
            # axes[0][i].grid(True)
            plot_fock_distribution(state_list[i], fig=fig, ax=axes[1][i])
            axes[2][i].plot(vec1, func1[i])
            axes[2][i].set_xticks(ticks)
            axes[2][i].set_xticklabels(labels, fontsize=20)
            axes[2][i].set_xlim(-xlim1, xlim1)
            axes[2][i].grid(True)
            axes[3][i].plot(vec2, func2[i])
            axes[3][i].set_xticks(ticks)
            axes[3][i].set_xticklabels(labels, fontsize=20)
            axes[3][i].set_xlim(-xlim2, xlim2)
            axes[3][i].grid(True)
            axes[4][i].plot(vec3, func3[i])
            axes[4][i].set_xticks(ticks)
            axes[4][i].set_xticklabels(labels, fontsize=20)
            axes[4][i].set_xlim(-xlim3, xlim3)
            axes[4][i].grid(True)

        print("plotting done, time: %6f" % float(time.time()-start))
        plt.show()
        exit()
