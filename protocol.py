# -*- coding: utf-8 -*-

"""
Provides an explicit simulation of the protocol given a feedback mode, many n_m

mode 0: Kitaev phase estimation. arxiv: quant-ph/9511026
mode 1: Adaptive repeated phase estimation PRA 63, 053804 (2001)
mode 2: Feedback off, Returns 0.
mode 3: returns a random number depending on m, x

More information can be found in arxiv: 1506.05033 (our paper) or my
M.Sc. thesis

The operations of a single round of the protocol are provided by
state_simulation.py
The feedback is provided by feedback.py

In addition to state_simulation, the module provides the simulation and
analysis of many measurement outcomes at a time.

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

import time
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
from scipy import integrate
import numpy as np

import feedback as fb
import state_simulation as sim
from qutip import *
import misc

np.set_printoptions(precision=2)        # Set numpy to print with 2 digits

N = 150                                 # dimension of Fock space
delta = 0.2                             # initial squeezing
r = np.log(1/delta)                     # computes the squeezing factor
mode = 2
n_points_plot = 1025
error_threshold = np.sqrt(np.pi)/6

N_p = 2
m = 2
Min = 0
Max = 2**m
# 2 ok

lim = 20
error_threshold = np.sqrt(np.pi)/6

# -----------------------------------------------------------------------------
# Experimental properties.                 All frequencies in GHz

wc = 8.2 * 2 * np.pi                       # bare freq of the cavity
wq = 7.46 * 2 * np.pi                      # bare freq of the qubit
# g = 0.01 * 2 * np.pi                     # can be used to set g
# chi = g**2/(np.abs(wc-wq))               # if g is set, use this to get chi
chi = 2.4/1000 * 2 * np.pi                 # chi
K_s = 3.61/1000000 * 2 * np.pi             # Kerr
K_qs = 4.2/1000000 * 2 * np.pi             # Qubit-dependent Kerr

# cavity operators
a = tensor(destroy(N), qeye(2))            # anihilation operator
nc = a.dag() * a                           # number operator
nnc = a.dag() * a.dag() * a * a            # Kerr operator

# qubit operators
b = tensor(qeye(N), destroy(2))            # anihilation operator
nq = b.dag() * b                           # number operator
sz = tensor(qeye(N), sigmaz())             # Pauli Z
sx = tensor(qeye(N), sigmax())             # Pauli X
hadamard = tensor(qeye(N), snot())         # Hadamard Gate

Hc = nc                                    # cavity evolution
H2c = nnc                                  # cavity Kerr
Hq = nq                                    # qubit evolution
Hqc = nc * sz                              # qubit-cavity
H2qc = nnc * sz

# Hamiltonian used in the protocol
H = - chi * Hqc     # - K_s * H2c - K_qs * H2qc

# -----------------------------------------------------------------------------


def final_state(n_m, H, psi0, mode):
    psi = psi0
    for j in np.arange(len(n_m)):
        psi = sim.perfect_protocol(psi, H, n_m, j, mode)
    # psi = sim.correcting_shift(n_m, psi, mode).ptrace(0)
    # x_vec = np.linspace(-lim,lim,n_points_plot)
    # dx = np.mean([np.diff(x_vec).max(), np.diff(x_vec).min()])
    # W, y_vec = wigner(psi.ptrace(0),x_vec, x_vec, method='fft')
    # pfunc = integrate.romb(W, dx = dx, axis = 1)
    # psi = sim.correcting_cheat(n_m, y_vec, pfunc, psi).ptrace(0)
    psi = sim.remove_qubit(psi)
    return psi


def result_range(Min, Max, m):
    """Return range min-max as bit strings of length m"""
    return np.arange(Min, Max)[:, np.newaxis] >> np.arange(m)[::-1] & 1

def result_random(N_results, m):
    """Return N_results random bit-strings of length m"""
    return np.random.random_integers(0,1,size=(N_results,m))

if __name__ == '__main__':
    start = time.time()
    print("initialized")

    psi0 = tensor(squeeze(N, r) * basis(N, 0), basis(2, 0)).unit()

    result_list = result_range(Min, Max, m)
    print(result_list)
    state_list = parallel_map(final_state, result_list,
                            task_args=(H, psi0, mode))
    print("obtained new state, time: %6f" % float(time.time()-start))
    x_vec, y_vec, W, qfunc, pfunc, qfunc_m, pfunc_m, p_z, p_x, error_rate =\
        sim.analysis_wrapper(state_list, lim, n_points_plot, error_threshold)

    print("error analysis done, time: %6f" % float(time.time()-start))
    print("p_z")
    print(p_z)
    print("p_x")
    print(p_x)
    print("error rate")
    print(np.asarray(error_rate))
    mins = np.asarray([np.mod(y_vec[np.argmax(pfunc[i])], np.sqrt(np.pi))
                       for i in np.arange(len(state_list))])
    print("mins")
    print(mins)

    inter_vec = np.linspace(-np.sqrt(np.pi)/2, np.sqrt(np.pi)/2, 200)
    vec = np.mod(y_vec+np.sqrt(np.pi)/2, np.sqrt(np.pi))-np.sqrt(np.pi)/2
    psi_noc = [np.abs(fb.psiv_noc(inter_vec, m, int(''.join(map(str, i)), 2),
                                  mode))**2 for i in result_list]
    psi_c = [np.abs(fb.psiv(inter_vec, m, int(''.join(map(str, i)), 2),
                            mode))**2 for i in result_list]
    pfunc_wrap = [misc.wrap_function(vec, pfunc[i], inter_vec)
                  for i in np.arange(len(pfunc))]

    func1 = psi_c
    vec1 = -inter_vec
    xlim1 = 0.5*np.sqrt(np.pi)

    func2 = psi_noc
    vec2 = -inter_vec
    xlim2 = 0.5*np.sqrt(np.pi)

    func3 = pfunc_wrap
    vec3 = inter_vec
    xlim3 = 0.5*np.sqrt(np.pi)

    wlim = parallel_map(sim.absmax, W)
    ticks = np.arange(-10*np.ceil(lim/np.sqrt(np.pi)),
                      10*np.ceil(lim/np.sqrt(np.pi)) + 1, 1) * np.sqrt(np.pi)
    labels = ["" for t in ticks]

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
