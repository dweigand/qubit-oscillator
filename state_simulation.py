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
import scipy.integrate as integrate
import numpy as np

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


"""
Comment to self:
fb.feedback and qubit_reset both use the measurement result of the current
round. That is not needed, and bad, if the measurement result is obtained
during the round, e.g. via RNG. Change that.

bad Idea to use x and n
 -> Seems necessary, as lists cannot be memoized efficiently
"""


# currently not used
def protocol(state, H, n_m, m, mode):
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
    # Workaround
    # used as workaround for the problems in protocol()
    x = int(''.join(map(str, n_m)), 2)
    #
    phi = fb.feedback(m+1, x, mode)
    state = hadamard * state
    result1 = controlled_rotation(state, H)
    state = result1.states[-1].unit()
    state = displace_(-1j * np.sqrt(np.pi/2)) * sx * state
    result2 = controlled_rotation(state, H)
    state = result2.states[-1].unit()
    state = hadamard * phasegate_(phi) * sx * state
    state = qubit_reset(state, n_m[m])
    return state.unit()


def perfect_protocol(state, H, n_m, m, mode):
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
    # Workaround
    # used as workaround for the problems in protocol()
    x = int(''.join(map(str, n_m)), 2)

    phi = fb.feedback(m+1, x, mode)
    state = hadamard * state
    state = controlled_zgate(displace(N, np.sqrt(np.pi/2))) * state
    state = hadamard * phasegate_(phi) * state
    state = qubit_reset(state, n_m[m])
    return state.unit()


def correcting_shift(n_m, state, mode):
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
    x = int(''.join(map(str, n_m)), 2)
    est = fb.estimate(len(n_m), x, mode)
    return (displace_(1j*est/(2*np.sqrt(2*np.pi))) * state).unit()


def correcting_cheat(n_m, y_vec, pfunc, state):
    """
    Correct the state based on an error analysis of pfunc

    Deprecated, should not be used. Uses lots of computational power and is not
    properly tested.

    Args:
        n_m: list of measurement results
        y_vec: x-vector for pfunc
        pfunc: wigner function in p-space
        state: state to be corrected
    Returns:
        array of length x_inter
    """
    y_vec = (np.mod(y_vec+np.sqrt(np.pi)/2,
                    np.sqrt(np.pi))-np.sqrt(np.pi)/2)*2*np.sqrt(np.pi)
    p = pfunc*np.exp(1j*y_vec)
    est = np.angle(integrate.simps(p, y_vec))
    print(n_m)
    print(est/(2*np.sqrt(np.pi)))
    return (displace_(-0) * state).unit()

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

    # Other methods for wigner() than 'fft' require huge storage
    # W = list(zip(*parallel_map(wigner, states, task_args = (x_vec, x_vec))))
    # yvecs = xvecs
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

    pfunc = integrate.romb(w, dx=dx, axis=1)
    qfunc = integrate.romb(w, dx=dy, axis=0)

    error = 1-integrate.romb(pfunc, dx=dy)
    if error > 0.001:
        print("rounding error of Wigner function is large: %6f" % error)

    pfunc_m = mask(pfunc, condlist_y)
    qfunc_m = mask(qfunc, condlist_x)
    p_z = integrate.romb(pfunc_m, dx=dy)
    p_x = integrate.romb(qfunc_m, dx=dx)
    error_rate = 1-p_x * p_z
    return qfunc, pfunc, qfunc_m, pfunc_m, p_z, p_x, error_rate


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


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# SIMULATION
if __name__ == '__main__':
    start = time.time()
    print("start")
    delta = 0.14                                # initial squeezing
    r = np.log(1/delta)                         # computes the squeezing factor

    # Hamiltonians
    Hc = nc                                     # cavity evolution
    H2c = nnc                                   # cavity Kerr
    Hq = nq                                     # qubit evolution
    Hqc = nc * sz                               # qubit-cavity
    H2qc = nnc * sz
    H = - chi * Hqc  # - K_s * H2c - K_qs * H2qc # Hamiltonian for protocol

    n_m = [0, 1]
    # n_2 = [1,1,1,1,0]]                        # measurement result
    mode = 1
    lim = (len(n_m)+2)*np.sqrt(np.pi)
    print("result is %s" % str(n_m))
    if mode == 0:
        print("feedback using rpe")
    elif mode == 1:
        print("feedback using arpe")
    elif mode == 2:
        print("feedback off")

    psi0 = tensor(squeeze(N, r) * basis(N, 0), basis(2, 0)).unit()
    psi = psi0
    print("evolving...")
    state_list = []
    # for m in np.arange(len(n_m)):
    #    [psi, result1, result2] = protocol(psi, n_m, m)
    #    state_list.append([psi.ptrace(0), result1, result2])

    for m in np.arange(len(n_m)):
        psi = protocol(psi, H, n_m, m, mode)
        state_list.append(psi.ptrace(0))

    psi_end = correcting_shift(n_m, psi, mode)
    print("obtained new state, time: %6f" % float(time.time()-start))

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Analysis

    states = [state_list[i] for i in np.arange(len(state_list))]
    states.insert(0, psi0.ptrace(0))
    states.append(psi_end.ptrace(0))

    global axes
    global fig
    fig, axes = plt.subplots(5, len(states), figsize=(12, 2))

    (x_vec, y_vec, W,
     qfunc, pfunc, qfunc_m, pfunc_m,
     p_z, p_x, error_rate) = analysis_wrapper(states, lim, n_points_plot,
                                              error_threshold)

    print(p_z)
    print(p_x)
    print(error_rate)
    func1 = qfunc
    vec1 = x_vec

    func2 = pfunc
    vec2 = y_vec

    func3 = pfunc_m
    vec3 = y_vec

    wlim = parallel_map(absmax, W)
    ticks = np.arange(-10*np.ceil(lim/np.sqrt(np.pi)),
                      10*np.ceil(lim/np.sqrt(np.pi)) + 1,
                      1) * np.sqrt(np.pi)/6
    xlim = np.sqrt(np.pi)
    labels = ["" for t in ticks]

    for i in np.arange(len(states)):
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
        plot_fock_distribution(states[i], fig=fig, ax=axes[1][i])
        axes[2][i].plot(vec1, func1[i])
        axes[2][i].set_xticks(ticks)
        axes[2][i].set_xticklabels(labels, fontsize=20)
        axes[2][i].set_xlim(-3*np.sqrt(np.pi), 3*np.sqrt(np.pi))
        axes[2][i].grid(True)
        axes[3][i].plot(vec2, func2[i])
        axes[3][i].set_xticks(ticks)
        axes[3][i].set_xticklabels(labels, fontsize=20)
        axes[3][i].set_xlim(-xlim, xlim)
        axes[3][i].grid(True)
        axes[4][i].plot(vec3, func3[i])
        axes[4][i].set_xticks(ticks)
        axes[4][i].set_xticklabels(labels, fontsize=20)
        axes[4][i].set_xlim(-xlim, xlim)
        axes[4][i].grid(True)

    print("result has been %s" % str(n_m))
    Pn_m = fb.P_no_error(len(n_m), int(''.join(map(str, n_m)), 2), mode)
    print("Z error rate should be %s" % str(float(1-Pn_m)))
    print("Z error rate is %s" % str(float(1-p_z[-1])))
    print("Total error rate is %s" % str(float(error_rate[-1])))
    print("done, time: %6f" % float(time.time()-start))
    plt.show()
