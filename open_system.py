#!/usr/bin/env python

""" Fouth order Suzuki-Trotter decomposition of the Liouvillian operator """
""" Computes things on the XXZ with boundary driving                     """

import numpy as np
from MPyS.MPS.InitialMPS import InitialMPS
from MPyS.MPO.XXZEvoMPO import XXZEvoMPO
from MPyS.MPO.XXZEvoMPOBoundDriv import XXZEvoMPOBoundDriv
from MPyS.Utils.GeneralUtils import expectation_value_trace
from MPyS.Utils.ReductionUtils import apply_mpo_variational
from MPyS.Utils.ReductionUtils import apply_mpo_svd
from MPyS.Utils.ReductionUtils import prepare

import time

np.random.seed(2)

sites = 4
phys_dim = 2
bond_dim = 100
epsilon = 1.0E-8
precision = 1.0E-12
tau = 0.5
delta = [0.5 for _ in range(sites)]
h_local = [0.0 for _ in range(sites)]
b_gamma = 1.0
mu = 0.01
dt = 0.1
max_variational = 3
pos = sites / 2

time1 = time.time()

# Initial state
IMps = InitialMPS(sites)

i_state = IMps.identity_state(phys_dim)

# Variational MPS state
#mps_struct = IMps.random_state(bond_dim, i_state[0].shape[2])
#mps_struct = prepare(mps_struct, sites)

# ST decomp times
dt_1 = dt / (4 - (4 ** (1.0 / 3.0)) )
dt_2 = dt_1
dt_3 = dt - (2.0 * dt_1) - (2.0 * dt_2)

# Liouville ST decomp MPOs
prec_mpo = 1.0E-14
eps_mpo = 1.0E-12
EvoMPO_t1 = XXZEvoMPOBoundDriv(sites, tau, delta, h_local, b_gamma, mu, dt_1, bond_dim, eps_mpo, prec_mpo)
EvoMPO_t3 = XXZEvoMPOBoundDriv(sites, tau, delta, h_local, b_gamma, mu, dt_3, bond_dim, eps_mpo, prec_mpo)

# Observables
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])
ident = np.identity(phys_dim)
ident_2 = np.identity(phys_dim ** 2)

# Spin magnetization in z direction
oset = [[ident_2 for _ in range(sites)] for _ in range(sites)]
for i in range(sites):
    oset[i][i] = np.kron(ident, sz)

# Current
jset_1 = [[ident_2 for _ in range(sites)] for _ in range(sites)]
jset_2 = [[ident_2 for _ in range(sites)] for _ in range(sites)]
for i in range(sites - 1):
    jset_1[i][i] = np.kron(ident, sx)
    jset_1[i][i + 1] = np.kron(ident, sy)
    jset_2[i][i] = np.kron(ident, sy)
    jset_2[i][i + 1] = np.kron(ident, sx)

#local_z, eva_z, norm_z = expectation_value_trace(i_state, oset, sites, sites)
#for i in range(sites):
#    print '0.0', i+1, local_z[i].real
#print

#local_j1, t_j1, n1 = expectation_value_trace(i_state, jset_1, sites, sites)
#local_j2, t_j2, n2 = expectation_value_trace(i_state, jset_2, sites, sites)
#for i in range(sites):
    #print '0.0', i+1, local_j1[i].real, local_j2[i].real, t_j1.real, t_j2.real
#    print '0.0', i+1, t_j1.real, t_j2.real, n2.real
#print

t = 0.0
for i in range(100):
    t = t + dt

    #mps_b = np.array(mps_struct, copy = True)
    mps_svd = apply_mpo_svd(EvoMPO_t1.MPO, i_state, sites, bond_dim, epsilon)
    i_state, k = apply_mpo_variational(i_state, mps_svd, EvoMPO_t1.MPO, sites,
            precision, max_variational)

    #mps_b = np.array(mps_struct, copy = True)
    mps_svd = apply_mpo_svd(EvoMPO_t1.MPO, i_state, sites, bond_dim, epsilon)
    i_state, k = apply_mpo_variational(i_state, mps_svd, EvoMPO_t1.MPO, sites,
            precision, max_variational)

    #mps_b = np.array(mps_struct, copy = True)
    mps_svd = apply_mpo_svd(EvoMPO_t3.MPO, i_state, sites, bond_dim, epsilon)
    i_state, k = apply_mpo_variational(i_state, mps_svd, EvoMPO_t3.MPO, sites,
            precision, max_variational)

    #mps_b = np.array(mps_struct, copy = True)
    mps_svd = apply_mpo_svd(EvoMPO_t1.MPO, i_state, sites, bond_dim, epsilon)
    i_state, k = apply_mpo_variational(i_state, mps_svd, EvoMPO_t1.MPO, sites,
            precision, max_variational)

    #mps_b = np.array(mps_struct, copy = True)
    mps_svd = apply_mpo_svd(EvoMPO_t1.MPO, i_state, sites, bond_dim, epsilon)
    i_state, k = apply_mpo_variational(i_state, mps_svd, EvoMPO_t1.MPO, sites,
            precision, max_variational)

    #local_z, eva_z, norm_z = expectation_value_trace(i_state, oset, sites, sites)
    #for i in range(sites):
    #    print t, i+1, local_z[i].real
    #print

    local_j1, t_j1, n1 = expectation_value_trace(i_state, jset_1, sites, sites)
    local_j2, t_j2, n2 = expectation_value_trace(i_state, jset_2, sites, sites)
    #for i in range(sites):
    #    print t, i+1, local_j1[i].real, local_j2[i].real, t_j1.real, t_j2.real, n1.real, n2.real
        #print t, i+1, t_j1.real, t_j2.real, n2.real
    #print

    print t, ((local_j1[pos] - local_j2[pos]) / mu).real

time2 = time.time()

print 'Time =', time2 - time1
