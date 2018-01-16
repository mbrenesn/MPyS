import numpy as np
from scipy.linalg import expm
from MPyS.Utils.GeneralUtils import svd
from MPyS.Utils.GeneralUtils import contract_tensors

class XXZEvoMPO:
    """ MPO Class for the even/odd separation of propagator   """
    """using the XXZ Hamiltonian for spin 1/2. (4th order ST) """
    def __init__(self, num_sites, tau, delta, h_local, dt):
        self.n = num_sites
        self.phys_dim = 2
        self.tau = tau
        self.delta = delta
        self.h_local = h_local
        self.dt = dt

        self.MPO_even_ = [0 for _ in range(num_sites)]
        self.MPO_odd_ = [0 for _ in range(num_sites)]

        self.construct_XXZEvoMPO_()

        self.MPO = self.mpo_mpo_contract_(self.MPO_odd_, self.MPO_even_)
        self.MPO = self.mpo_mpo_contract_(self.MPO_even_, self.MPO)

    def construct_XXZEvoMPO_(self):
        sx = np.array([[0.0, 1.0], [1.0, 0.0]])
        sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
        sz = np.array([[1.0, 0.0], [0.0, -1.0]])
        ident = np.identity(self.phys_dim)

        t_ident = np.reshape(ident, (1, 1, self.phys_dim, self.phys_dim), order = 'F')
        for i in range(self.n):
            self.MPO_even_[i] = t_ident
            self.MPO_odd_[i] = t_ident

        for i in range(0, self.n - 1, 2):
            h = self.tau * ( np.kron(sx, sx) + np.kron(sy, sy) + (self.delta[i] * np.kron(sz, sz))
                + ((self.h_local[i] / 2) * np.kron(sz, ident)) +
                    ((self.h_local[i + 1] / 2) * np.kron(ident, sz)))

            w = expm(-1.0j * self.dt * h)

            U, V = self.bond_MPO_svd_(w)

            self.MPO_odd_[i] = U
            self.MPO_odd_[i + 1] = V

        for i in range(1, self.n - 1, 2):
            h = self.tau * ( np.kron(sx, sx) + np.kron(sy, sy) + (self.delta[i] * np.kron(sz, sz))
                + ((self.h_local[i] / 2) * np.kron(sz, ident)) +
                    ((self.h_local[i + 1] / 2) * np.kron(ident, sz)))

            w = expm(-0.5j * self.dt * h)

            U, V = self.bond_MPO_svd_(w)

            self.MPO_even_[i] = U
            self.MPO_even_[i + 1] = V

    def bond_MPO_svd_(self, w):
        w = np.reshape(w, (self.phys_dim, self.phys_dim, self.phys_dim, self.phys_dim), order = 'F')
        w = np.transpose(w, (3, 1, 2, 0))
        w = np.reshape(w, (self.phys_dim * self.phys_dim, self.phys_dim * self.phys_dim), order = 'F')

        U, S, V = svd(w)
        eta = S.shape[0]
        Smat = np.zeros((eta, eta))
        np.fill_diagonal(Smat, S)

        U = U.dot(np.sqrt(Smat))
        V = np.sqrt(Smat).dot(V)
        U = np.reshape(U, (self.phys_dim, self.phys_dim, eta), order = 'F')
        U = np.expand_dims(U, axis = 3)
        U = np.transpose(U, (3, 2, 1, 0))
        V = np.reshape(V, (eta, self.phys_dim, self.phys_dim), order = 'F')
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
