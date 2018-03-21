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

import argparse
import sys

# Parsing arguments
parser = argparse.ArgumentParser(description='Calculate the current in the \
	NESS using MPOs 4th order Trotter')

parser.add_argument('--l', action='store', dest='l', type=int,
                    help='[INT] Number of sites')
parser.add_argument('--alpha', action='store', dest='alpha', type=float,
                    help='[FLOAT] Hamiltonian parameter')
parser.add_argument('--delta', action='store', dest='delta', type=float,
                    help='[FLOAT] Hamiltonian parameter')
parser.add_argument('--h', action='store', type=float, dest='h',
                    help='[FLOAT] Impurity strenght in the middle of the chain')
parser.add_argument('--dt', action='store', type=float, dest='dt',
                    help='[FLOAT] Timestep for 4th order ST')
parser.add_argument('--mu', action='store', type=float, dest='mu',
                    help='[FLOAT] Coupling constant')
parser.add_argument('--boundary_gamma', action='store', type=float, dest='b_gamma',
                    default=None, help='[FLOAT] Driving parameter, defaults to None')
parser.add_argument('--bond_dim', action='store', type=int, dest='bond_dim',
                    help='[INT] Maximum bond dimension to use')

args = parser.parse_args()

# Hamiltonian arguments
sites = args.l
alpha_val = args.alpha
delta_val = args.delta
h = args.h

phys_dim = 2
bond_dim = args.bond_dim
epsilon = 1.0E-8
precision = 1.0E-12
pos = sites / 2
alpha = [alpha_val for _ in range(sites)]
delta = [delta_val for _ in range(sites)]
h_local = [0.0 for _ in range(sites)]
for i in range(0, sites, 2):
    h_local[i] = -1.0 * h
b_gamma = args.b_gamma
mu = args.mu
dt = args.dt
max_variational = 3
steps = int(5.0 * (sites / dt))

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

print '# L =', sites
print '# alpha =', alpha
print '# delta =', delta
print '# h =', h_local
print '# mu =', mu
print '# b_gamma =', b_gamma
print '# dt =', dt
print '# chi =', bond_dim
sys.stdout.flush()

name = ('magnetisation_' + 'l' + str(sites) + '_delta' + str(delta_val)
        + '_h' + str(h) + '_mu' + str(mu) + '.dat')
bufsize = 1
f1 = open('%s' % name, 'w+', bufsize)
print >> f1, '# L =', sites
print >> f1, '# alpha =', alpha
print >> f1, '# delta =', delta
print >> f1, '# h =', h_local
print >> f1, '# mu =', mu
print >> f1, '# b_gamma =', b_gamma
print >> f1, '# dt =', dt
print >> f1, '# chi =', bond_dim

# Liouville ST decomp MPOs
prec_mpo = 1.0E-14
eps_mpo = 1.0E-12
EvoMPO_t1 = XXZEvoMPOBoundDriv(sites, alpha, delta, h_local,
        b_gamma, mu, dt_1, bond_dim, eps_mpo, prec_mpo)
EvoMPO_t3 = XXZEvoMPOBoundDriv(sites, alpha, delta, h_local,
        b_gamma, mu, dt_3, bond_dim, eps_mpo, prec_mpo)

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

# Spin
jset_1 = [ident_2 for _ in range(sites)]
jset_2 = [ident_2 for _ in range(sites)]
jset_1[pos] = alpha[pos] * np.kron(ident, sx)
jset_1[pos + 1] = np.kron(ident, sy)
jset_2[pos] = alpha[pos] * np.kron(ident, sy)
jset_2[pos + 1] = np.kron(ident, sx)

# Initial expectation values

# Spin
#local_z, eva_z, norm_z = expectation_value_trace(i_state, oset, sites, sites)
#for i in range(sites):
#    print >> f1, '0.0', i+1, local_z[i].real / mu
#print >> f1, ''

# Spin Current
local_j1, t_j1, n1 = expectation_value_trace(i_state, jset_1, sites, 1)
local_j2, t_j2, n2 = expectation_value_trace(i_state, jset_2, sites, 1)
print '# Time', 'Current'
print '0.0', ((local_j1[0] - local_j2[0]) / mu).real
sys.stdout.flush()

t = 0.0
for i in range(steps):
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

    if(i % 100 == 0):
        local_z, eva_z, norm_z = expectation_value_trace(i_state, oset, sites, sites)
        for i in range(sites):
            print >> f1, t, i+1, local_z[i].real / mu
        print >> f1, ''

    local_j1, t_j1, n1 = expectation_value_trace(i_state, jset_1, sites, 1)
    local_j2, t_j2, n2 = expectation_value_trace(i_state, jset_2, sites, 1)

    print t, ((local_j1[0] - local_j2[0]) / mu).real
    sys.stdout.flush()

time2 = time.time()
print '# Time =', time2 - time1
sys.stdout.flush()
f1.close()
