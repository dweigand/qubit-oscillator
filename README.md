# qubit-oscillator
This is a suite to simulate the protocol for encoding qubits in oscillators proposed in arxiv/1506.05033

The paper that most of this work is based on can be found here: http://arxiv.org/pdf/1506.05033v3.pdf
Of special interest are Figs. 1,2,5,10 and 15. Sec 1C introduces the shift basis used for the analysis of code states.
A brief explanation of the algorithms used for encoding can be found in sections 2b,c. Section 2e gives details on the analysis of states, including the analytic result for the wave-function in v-space, which is used in feedback.py .
A brief derivation of arpe can be found in appendix C

## Hirarchy
The hirarchy of modules is
misc.py
feedback.py (used to create fig 15 of the paper)
state_simulation.py
protocol.py

## Current configuration
To simplify finding issues, the protocol currently simulates a perfect circuit, the functions for Hamiltonian evolution are not used.

## Output if the files are called as main:
the most interesting tool for the analysis of multiple runs of the protocol is protocol.py
If called as main, it shows a plot of N_result columns, where N_result is the number of results analyzed. Feedback mode, measurement results etc. can be changed at the top of the file.
Each column corresponds to one run of the protocol defined by a measurement result, all columns use the same feedback method.
there are 5 rows, from top to bottom:
Wigner function of the cavity state, Cavity state in Fock space (mostly there to check if it is chosen large enough), corrected analytic wave function (to show that the analytic results are self-consistent), uncorrected analytic wave function, simulated wave function.

The last two should be identical except at the edges of the plot, where the simulated wave function is underestimated (see doc of misc.py)
