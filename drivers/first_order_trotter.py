#!/usr/bin/env python

import numpy as np
from MPyS.MPS.InitialMPS import InitialMPS
from MPyS.MPO.XXZEvoMPO import XXZEvoMPO
from MPyS.Utils.GeneralUtils import expectation_value
from MPyS.Utils.ReductionUtils import apply_mpo_variational

np.random.seed(2)

sites = 50
phys_dim = 2
bond_dim = 10
tau = 1.0
delta = [1.0 for _ in range(sites)]
h_local = [0.0 for _ in range(sites)]
dt_even = 0.01
dt_odd = 0.01
max_variational = 5

flipped = sites / 2

IMps = InitialMPS(sites)

i_state = IMps.single_flip(flipped)

EvoMPO = XXZEvoMPO(sites, tau, delta, h_local, dt_even, dt_odd)

# Observables
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])
ident = np.identity(phys_dim)

oset = [ident for _ in range(sites)]
oset[flipped] = sz

# Careful, this is starting the variational algorithm with the same state
for i in range(50):
    i_state, k = apply_mpo_variational(i_state, EvoMPO.MPO_even, sites, bond_dim,
            1.0e-07, max_variational)
    i_state, k = apply_mpo_variational(i_state, EvoMPO.MPO_odd, sites, bond_dim,
            1.0e-07, max_variational)
    local, eva, norm = expectation_value(i_state, oset, sites, 1)
    print eva.real, norm.real
