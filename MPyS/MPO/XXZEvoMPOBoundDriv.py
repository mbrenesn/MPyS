import numpy as np
from scipy.linalg import expm
from MPyS.Utils.GeneralUtils import svd
from MPyS.Utils.GeneralUtils import contract_tensors
from MPyS.Utils.ReductionUtils import prepare_decimate
from MPyS.Utils.ReductionUtils import apply_mpo_variational

class XXZEvoMPOBoundDriv:
    """ MPO Class for the even/odd separation of propagator   """
    """using the XXZ Hamiltonian for spin 1/2. (4th order ST) """
    def __init__(self, num_sites, tau, delta, h_local, boundary_gamma, mu, dt, d_max, epsilon, precision):
        self.n = num_sites
        self.phys_dim = 2
        self.phys_ext_dim = 4
        self.tau = tau
        self.delta = delta
        self.h_local = h_local
        self.b_gamma = boundary_gamma
        self.mu = mu
        self.dt = dt
        self.d_max = d_max
        self.epsilon = epsilon
        self.precision = precision
        self.max = 5

        self.MPO_even_ = [0 for _ in range(num_sites)]
        self.MPO_odd_ = [0 for _ in range(num_sites)]
        self.vec_MPO_ = [0 for _ in range(num_sites)]

        self.construct_XXZEvoMPO_()

        self.MPO = self.mpo_mpo_contract_(self.MPO_odd_, self.MPO_even_)
        self.MPO = self.mpo_mpo_contract_(self.MPO_even_, self.MPO)

        self.vec_mpo_()

        temp = [0 for _ in range(num_sites)]
        for i in range(num_sites):
            temp[i] = np.copy(self.vec_MPO_[i])

        self.MPO = self.mps_trunc_svd_(temp)
        self.MPO = self.mps_var_compress_(self.vec_MPO_, self.MPO)

        self.mat_mps_()

    def construct_XXZEvoMPO_(self):
        sx = np.array([[0.0, 1.0], [1.0, 0.0]])
        sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
        sz = np.array([[1.0, 0.0], [0.0, -1.0]])
        s = np.array([[0.0, 0.0], [1.0, 0.0]])
        s_t = np.array([[0.0, 1.0], [0.0, 0.0]])
        ident = np.identity(self.phys_dim)
        ident_2 = np.identity(self.phys_ext_dim)

        h_1 = ((-0.5j * self.h_local[0]) * np.kron( np.kron(ident, sz), ident_2 ) +
              ( 0.5j * self.h_local[0]) * np.kron( np.kron(sz, ident), ident_2 ))
        h_n = ((-0.5j * self.h_local[self.n - 1]) * np.kron( ident_2, np.kron(ident, sz) ) +
              ( 0.5j * self.h_local[self.n - 1]) * np.kron( ident_2, np.kron(sz, ident) ))

        j = np.sqrt( self.b_gamma * (1 + self.mu) * s_t )
        l_1 = np.kron( 2.0 * np.kron( j.conjugate(), j )
                       - (np.kron( ident, j.transpose().dot(j) )
                       + np.kron( (j.transpose().dot(j)).transpose(), ident) ), ident_2)
        j = np.sqrt( self.b_gamma * (1 - self.mu) * s )
        l_2 = np.kron( 2.0 * np.kron( j.conjugate(), j )
                       - (np.kron( ident, j.transpose().dot(j) )
                       + np.kron( (j.transpose().dot(j)).transpose(), ident) ), ident_2)

        j = np.sqrt( self.b_gamma * (1 + self.mu) * s )
        ln_1 = np.kron ( ident_2, 2.0 * np.kron( j.conjugate(), j )
                                  - (np.kron( ident, j.transpose().dot(j) )
                                  + np.kron( (j.transpose().dot(j)).transpose(), ident ) ) )
        j = np.sqrt( self.b_gamma * (1 - self.mu) * s_t )
        ln_2 = np.kron ( ident_2, 2.0 * np.kron( j.conjugate(), j )
                                  - (np.kron( ident, j.transpose().dot(j) )
                                  + np.kron( (j.transpose().dot(j)).transpose(), ident ) ) )

        t_ident = np.reshape(ident_2, (1, 1, self.phys_ext_dim, self.phys_ext_dim), order = 'F')
        for i in range(self.n):
            self.MPO_even_[i] = t_ident
            self.MPO_odd_[i] = t_ident

        for i in range(1, self.n, 2):
            h = self.tau * ( np.kron( np.kron(ident, sx), np.kron(ident, sx) )
                           + np.kron( np.kron(ident, sy), np.kron(ident, sy) )
                           + (self.delta[i - 1] * np.kron( np.kron(ident, sz), np.kron(ident,sz) ))
                + ( (self.h_local[i - 1] / 2.0) * np.kron( np.kron(ident, sz), ident_2 ) )
                + ( (self.h_local[i] / 2.0) * np.kron( ident_2, np.kron(ident, sz) ) ) )

            l_h = -1.0j * h

            h = self.tau * ( np.kron( np.kron(sx.transpose(), ident), np.kron(sx.transpose(), ident) )
                           + np.kron( np.kron(sy.transpose(), ident), np.kron(sy.transpose(), ident) )
                           + (self.delta[i - 1] * np.kron( np.kron(sz.transpose(), ident),
                                                           np.kron(sz.transpose(), ident) ))
                + ( (self.h_local[i - 1] / 2.0) * np.kron( np.kron(sz, ident), ident_2 ) )
                + ( (self.h_local[i] / 2.0) * np.kron( ident_2, np.kron(sz, ident) ) ) )

            l_h = l_h + (1.0j * h)

            w = expm(self.dt * l_h)

            if(i == 1):
                w = expm(self.dt * (l_h + h_1 + l_1 + l_2))
            elif(i == (self.n - 1)):
                w = expm(self.dt * (l_h + h_n + ln_1 + ln_2))

            U, V = self.bond_MPO_svd_(w, self.phys_ext_dim)

            self.MPO_odd_[i - 1] = U
            self.MPO_odd_[i] = V

        for i in range(2, self.n, 2):
            h = self.tau * ( np.kron( np.kron(ident, sx), np.kron(ident, sx) )
                           + np.kron( np.kron(ident, sy), np.kron(ident, sy) )
                           + (self.delta[i - 1] * np.kron( np.kron(ident, sz), np.kron(ident,sz) ))
                + ( (self.h_local[i - 1] / 2.0) * np.kron( np.kron(ident, sz), ident_2 ) )
                + ( (self.h_local[i] / 2.0) * np.kron( ident_2, np.kron(ident, sz) ) ) )

            l_h = -1.0j * h

            h = self.tau * ( np.kron( np.kron(sx.transpose(), ident), np.kron(sx.transpose(), ident) )
                           + np.kron( np.kron(sy.transpose(), ident), np.kron(sy.transpose(), ident) )
                           + (self.delta[i - 1] * np.kron( np.kron(sz.transpose(), ident),
                                                           np.kron(sz.transpose(), ident) ))
                + ( (self.h_local[i - 1] / 2.0) * np.kron( np.kron(sz, ident), ident_2 ) )
                + ( (self.h_local[i] / 2.0) * np.kron( ident_2, np.kron(sz, ident) ) ) )

            l_h = l_h + (1.0j * h)

            w = expm(0.5 * self.dt * l_h)

            if(i == (self.n - 1)):
                w = expm(0.5 * self.dt * (l_h + h_n + ln_1 + ln_2))

            U, V = self.bond_MPO_svd_(w, self.phys_ext_dim)

            self.MPO_even_[i - 1] = U
            self.MPO_even_[i] = V

    def bond_MPO_svd_(self, w, dim):
        w = np.reshape(w, (dim, dim, dim, dim), order = 'F')
        w = np.transpose(w, (3, 1, 2, 0))
        w = np.reshape(w, (dim * dim, dim * dim), order = 'F')

        U, S, V = svd(w)
        eta = S.shape[0]
        Smat = np.zeros((eta, eta))
        np.fill_diagonal(Smat, S)

        U = U.dot(np.sqrt(Smat))
        V = np.sqrt(Smat).dot(V)
        U = np.reshape(U, (dim, dim, eta), order = 'F')
        U = np.expand_dims(U, axis = 3)
        U = np.transpose(U, (3, 2, 1, 0))
        V = np.reshape(V, (eta, dim, dim), order = 'F')
        V = np.expand_dims(V, axis = 3)
        V = np.transpose(V, (0, 3, 2, 1))

        return U, V

    def mpo_mpo_contract_(self, mpo_1, mpo_2):
        mpo = [0 for _ in range(self.n)]
        for i in range(self.n):
            ten, numindx = contract_tensors(mpo_1[i], 4, (2, ), mpo_2[i], 4, (3, ))
            s_ten = ten.shape
            ten = np.transpose(ten, (3, 0, 4, 1, 5, 2))
            mpo[i] = np.reshape(ten, ( s_ten[0]*s_ten[3], s_ten[1]*s_ten[4], s_ten[5], s_ten[2] ))

        return mpo

    def vec_mpo_(self):
        for i in range(self.n):
            s_a = self.MPO[i].shape
            self.vec_MPO_[i] = np.reshape(self.MPO[i], [ s_a[0], s_a[1], s_a[2] * s_a[3] ])

    def mat_mps_(self):
        for i in range(self.n):
            s_a = self.MPO[i].shape
            self.MPO[i] = np.reshape(self.MPO[i], [ s_a[0], s_a[1],
                int(np.sqrt(s_a[2])), int(np.sqrt(s_a[2])) ])

    def mps_trunc_svd_(self, mps):
        for i in range(self.n - 1, 0, -1):
            mps[i], u, db = prepare_decimate(mps[i], self.d_max, self.epsilon, 'rl')
            mps[i - 1], nm = contract_tensors(mps[i - 1], 3, (1, ), u, 2, (0, ))
            mps[i - 1] = np.transpose(mps[i - 1], (0, 2, 1))

        return mps

    def mps_var_compress_(self, mps, mps_svd):
        d = self.MPO[0].shape[2]
        ident = [0 for _ in range(self.n)]
        x = np.identity(d)
        x = np.reshape(x, (1, 1, d, d))
        for i in range(self.n):
            ident[i] = x
        mps, k = apply_mpo_variational(mps, mps_svd, ident, self.n, self.precision, self.max)

        return mps
