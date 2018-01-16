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

