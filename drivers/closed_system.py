#!/usr/bin/env python

""" Fouth order Suzuki-Trotter decomposition of the Liouvillian operator """
""" Computes things on the XXZ with boundary driving                     """

import numpy as np
from MPyS.MPS.InitialMPS import InitialMPS
from MPyS.MPO.XXZEvoMPO import XXZEvoMPO
from MPyS.Utils.GeneralUtils import expectation_value
from MPyS.Utils.ReductionUtils import apply_mpo_variational

np.random.seed(2)

sites = 8
phys_dim = 2
bond_dim = 60
tau = 1.0
delta = [1.0 for _ in range(sites)]
h_local = [0.0 for _ in range(sites)]
b_gamma = 1.0
mu = 0.75
dt = 0.1
max_variational = 5

IMps = InitialMPS(sites)

i_state = IMps.domain_wall_state()

dt_1 = dt / (4 - (4 ** (1.0 / 3.0)) )
dt_2 = dt_1
dt_3 = dt - (2.0 * dt_1) - (2.0 * dt_2)

EvoMPO_t1 = XXZEvoMPO(sites, tau, delta, h_local, dt_1)
EvoMPO_t3 = XXZEvoMPO(sites, tau, delta, h_local, dt_3)

# Observables
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])
ident = np.identity(phys_dim)

oset = [[ident for _ in range(sites)] for _ in range(sites)]
for i in range(sites):
    oset[i][i] = sz

# Careful, this is starting the variational algorithm with the same state
local, eva, norm = expectation_value(i_state, oset, sites, sites)
for i in range(sites):
    print '0.0', i+1, local[i].real
print
t = 0.0
for i in range(10):
    t = t + dt

    i_state, k = apply_mpo_variational(i_state, EvoMPO_t1.MPO, sites, bond_dim,
            1.0e-07, max_variational)

    i_state, k = apply_mpo_variational(i_state, EvoMPO_t1.MPO, sites, bond_dim,
            1.0e-07, max_variational)

    i_state, k = apply_mpo_variational(i_state, EvoMPO_t3.MPO, sites, bond_dim,
            1.0e-07, max_variational)

    i_state, k = apply_mpo_variational(i_state, EvoMPO_t1.MPO, sites, bond_dim,
            1.0e-07, max_variational)

    i_state, k = apply_mpo_variational(i_state, EvoMPO_t1.MPO, sites, bond_dim,
            1.0e-07, max_variational)

    local, eva, norm = expectation_value(i_state, oset, sites, sites)
    for i in range(sites):
        print t, i+1, local[i].real
    print
