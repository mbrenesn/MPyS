import numpy as np

class InitialMPS:
    """ MPS Class for spin 1/2 systems"""
    def __init__(self, num_sites):
        self.n = num_sites
        self.down_ = np.array([1, 0])
        self.up_ = np.array([0, 1])

    def single_flip(self, j):
        state = [0 for _ in range(self.n)]
        for i in range(self.n):
            state[i] = np.reshape(self.up_, (1, 1, 2), order = 'F')
            if(i == j):
                state[i] = np.reshape(self.down_, (1, 1, 2), order = 'F')

        return state

    def domain_wall_state(self):
        state = [0 for _ in range(self.n)]
        for i in range(self.n / 2):
            state[i] = np.reshape(self.down_, (1, 1, 2), order = 'F')
        for i in range(self.n / 2, self.n):
            state[i] = np.reshape(self.up_, (1, 1, 2), order = 'F')

        return state

    def neel_state(self):
        state = [0 for _ in range(self.n)]
        for i in range(self.n):
            if i % 2 == 0:
                state[i] = np.reshape(self.down_, (1, 1, 2), order = 'F')
            else:
                state[i] = np.reshape(self.up_, (1, 1, 2), order = 'F')

        return state

    def random_state(self, bond_dim, phys_dim):
        state = [0 for _ in range(self.n)]

        state[0] = (1.0 / np.sqrt(bond_dim)) * np.random.randn(1, bond_dim, phys_dim)
        state[self.n - 1] = (1.0 / np.sqrt(bond_dim)) * np.random.randn(bond_dim, 1, phys_dim)
        for i in range(1, self.n - 1):
            state[i] = (1.0 / np.sqrt(bond_dim)) * np.random.randn(bond_dim, bond_dim, phys_dim)

        return state

    def identity_state(self, phys_dim):
        phys_ext_dim = phys_dim ** 2
        state = [0 for _ in range(self.n)]
        ident = np.identity(phys_dim)
        ident = ident / (np.trace(ident))
        ident = np.reshape(ident, (phys_ext_dim, 1))
        ident = np.reshape(ident, (1, 1, phys_ext_dim))
        for i in range(self.n):
            state[i] = ident

        return state

    def ramp_state(self, phys_dim, mu):
        phys_ext_dim = phys_dim ** 2
        state = [0 for _ in range(self.n)]
        mu = mu / 2.0
        for i in range(1, self.n + 1):
            val = ( ((-2.0 * mu) * (i - 1)) / (self.n - 1) ) + mu
            stl = np.zeros((2, 2))
            stl[0, 0] = 0.5 + val
            stl[1, 1] = 0.5 - val
            stl = np.reshape(stl, (phys_ext_dim, 1))
            stl = np.reshape(stl, (1, 1, phys_ext_dim))
            state[i - 1] = stl

        return state
"""
    def partial_ramp_state(self, phys_dim, mu, dop):
        phys_ext_dim = phys_dim ** 2
        state = [0 for _ in range(self.n)]
        mu = mu / 2.0
        val = mu / 2.0
        st_p = np.array([[(0.5 + val), 0.0], [0.0, (0.5 - val)]])
        st_m = np.array([[(0.5 - val), 0.0], [0.0, (0.5 + val)]])
        st_p = np.reshape(st_p, (phys_ext_dim, 1))
        st_m = np.reshape(st_m, (phys_ext_dim, 1))
        st_p = np.reshape(st_p, (1, 1, phys_ext_dim))
        st_m = np.reshape(st_m, (1, 1, phys_ext_dim))
        init = 2 * dop
        final_half = init + dop
        final = init + (2 * dop)
        for i in range(init - 1):
            state[i] = st_p
        val = (mu / 2.0)
        for i in range(init - 1, final_half - 1):
            val = val - ( (mu / 4.0) / dop )
            stl_p = np.zeros((2, 2))
            stl_m = np.zeros((2, 2))
            stl_p[0, 0] = 0.5 + val
            stl_p[1, 1] = 0.5 - val
            stl_m[0, 0] = 0.5 - val
            stl_m[1, 1] = 0.5 + val
            stl_p = np.reshape(stl_p, (phys_ext_dim, 1))
            stl_m = np.reshape(stl_m, (phys_ext_dim, 1))
            stl_p = np.reshape(stl_p, (1, 1, phys_ext_dim))
            stl_m = np.reshape(stl_m, (1, 1, phys_ext_dim))
            state[i] = stl_p
            state[self.n - i - 1] = stl_m
        for i in range(final - 1, self.n):
            state[i] = st_m

        return state
"""
