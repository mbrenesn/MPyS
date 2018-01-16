import numpy as np
import sys
from MPyS.MPS.InitialMPS import InitialMPS
from MPyS.Utils.GeneralUtils import svd
from MPyS.Utils.GeneralUtils import update_c_right
from MPyS.Utils.GeneralUtils import update_c_left
from MPyS.Utils.GeneralUtils import contract_tensors

def apply_mpo_variational(mps_a, mps_struct, mpo_x, n, precision, max_variational):
    c_storage = init_c_storage(mps_struct, mpo_x, mps_a, n)

    count = 0
    while(True):
        count = count + 1
        k_values = []

        # First cycle from i=1 to i=N-1
        for j in range(n - 1):
            tensor_b, k = reduceD_onesite(mps_a[j], mpo_x[j], c_storage[j], c_storage[j + 1])
            tensor_b, u, db = prepare_one_site(tensor_b, 'lr')
            mps_struct[j] = tensor_b
            k_values.append(k)

            # Update
            c_storage[j + 1] = update_c_left(c_storage[j], tensor_b, mpo_x[j], mps_a[j])

        for j in range(n - 1, 0, -1):
            tensor_b, k = reduceD_onesite(mps_a[j], mpo_x[j], c_storage[j], c_storage[j + 1])
            tensor_b, u, db = prepare_one_site(tensor_b, 'rl')
            mps_struct[j] = tensor_b
            k_values.append(k)

            # Update
            c_storage[j] = update_c_right(c_storage[j + 1], tensor_b, mpo_x[j], mps_a[j])

        if( (np.std(k_values) / np.abs(np.mean(k_values))) < precision or count == max_variational ):
            mps_struct[0], numind_x = contract_tensors(mps_struct[0], 3, (1, ), u, 2, (0, ))
            mps_struct[0] = np.transpose(mps_struct[0], (0, 2, 1))
            break

    return mps_struct, k

def prepare(mps, n):
    for i in range(n - 1, 0, -1):
        mps[i], u, db = prepare_one_site(mps[i], 'rl')
        mps[i - 1], numind_x = contract_tensors(mps[i - 1], 3, (1, ), u, 2, (0, ))
        mps[i - 1] = np.transpose(mps[i - 1], (0, 2, 1))

    b = np.reshape(mps[0], [np.prod(np.array(mps[0].shape)), 1], order = 'F')
    k = (b.conjugate().transpose().dot(b))
    mps[0] = mps[0] / np.sqrt(k)

    return mps

def prepare_one_site(a, direction):
    (bond_dim_1, bond_dim_2, phys_dim) = a.shape

    if(direction == 'lr'):
        a = np.transpose(a, (2, 0, 1))
        a = np.reshape(a, [phys_dim * bond_dim_1, bond_dim_2], order = 'F')
        b, s, u = svd(a)
        dimension_bond = s.shape[0]
        smat = np.zeros((dimension_bond, dimension_bond))
        np.fill_diagonal(smat, s)
        b = np.reshape(b, [phys_dim, bond_dim_1, dimension_bond], order = 'F')
        b = np.transpose(b, (1, 2, 0))
        u = smat.dot(u)
    elif(direction == 'rl'):
        a = np.transpose(a, (0, 2, 1))
        a = np.reshape(a, [bond_dim_1, phys_dim * bond_dim_2], order = 'F')
        u, s , b = svd(a)
        dimension_bond = s.shape[0]
        smat = np.zeros((dimension_bond, dimension_bond))
        np.fill_diagonal(smat, s)
        b = np.reshape(b, [dimension_bond, phys_dim, bond_dim_2], order = 'F')
        b = np.transpose(b, (0, 2, 1))
        u = u.dot(smat)
    else:
        print 'Unvalid argument to preprare_one_site()'
        sys.exit()

    return b, u, dimension_bond

def init_c_storage(mps_b, mpo_x, mps_a, n):
    c_storage = [np.array([[0]]) for _ in range(n + 1)]
    c_storage[0] = np.array([[1.0]])
    c_storage[n] = np.array([[1.0]])
    for i in range(n - 1, 0, -1):
        c_storage[i] = update_c_right(c_storage[i + 1], mps_b[i], mpo_x[i], mps_a[i])

    return c_storage

def reduceD_onesite(tensor_a, tensor_x, c_left, c_right):
    c_left, numind_x = contract_tensors(c_left, 3, (2, ), tensor_a, 3, (0, ))
    c_left, numind_x = contract_tensors(c_left, 4, (1, 3), tensor_x, 4, (0, 3))

    tensor_b, numind_x = contract_tensors(c_left, 4, (2, 1), c_right, 3, (1, 2))
    if(len(tensor_b.shape) == 2):
        tensor_b = np.expand_dims(tensor_b, axis = -1)
    tensor_b = np.transpose(tensor_b, (0, 2, 1))

    b = np.reshape(tensor_b, [np.prod(np.array(tensor_b.shape)), 1], order = 'F')
    k = (b.conjugate().transpose().dot(b))

    return tensor_b, k

def apply_mpo_svd(mpo_x, mps_a, n, d_max, epsilon):
    mps = [0 for _ in range(n)]
    a, nm = contract_tensors(mps_a[0], 3, (2, ), mpo_x[0], 4, (3, ))
    s_a = a.shape
    mps[0] = np.transpose(a, (2, 0, 3, 1, 4))
    mps[0] = np.reshape(mps[0], [s_a[0] * s_a[2], s_a[1] * s_a[3], s_a[4]])
    for i in range(n - 1):
        mps[i], u, db = prepare_decimate(mps[i], d_max, epsilon, 'lr')
        a, nm = contract_tensors(mps_a[i + 1], 3, (2, ), mpo_x[i + 1], 4, (3, ))
        s_a = a.shape
        mps[i + 1] = np.transpose(a, (2, 0, 3, 1, 4))
        mps[i + 1] = np.reshape(mps[i + 1], [s_a[0] * s_a[2], s_a[1] * s_a[3], s_a[4]])
        mps[i + 1], nm = contract_tensors(u, 2, (1, ), mps[i + 1], 3, (0, ))

    return mps

def prepare_decimate(a, d_max, epsilon, direction):
    (bond_dim_1, bond_dim_2, phys_dim) = a.shape

    if(direction == 'lr'):
        a = np.transpose(a, (2, 0, 1))
        a = np.reshape(a, [phys_dim * bond_dim_1, bond_dim_2], order = 'F')
        b, s, u = svd(a)
        v_lam = np.power(s, 2)
        v_lam = v_lam / np.sum(v_lam)

        trunc_d = np.argmin( np.abs( np.cumsum(v_lam) - (1 - epsilon) ) ) + 1
        if trunc_d > d_max:
            trunc_d = d_max
        #Truncation
        u = u[:trunc_d, :]
        s = s[:trunc_d]
        b = b[:, :trunc_d]

        dimension_bond = s.shape[0]
        smat = np.zeros((dimension_bond, dimension_bond))
        np.fill_diagonal(smat, s)
        b = np.reshape(b, [phys_dim, bond_dim_1, dimension_bond], order = 'F')
        b = np.transpose(b, (1, 2, 0))
        u = smat.dot(u)
    elif(direction == 'rl'):
        a = np.transpose(a, (0, 2, 1))
        a = np.reshape(a, [bond_dim_1, phys_dim * bond_dim_2], order = 'F')
        u, s , b = svd(a)
        v_lam = np.power(s, 2)
        v_lam = v_lam / np.sum(v_lam)

        trunc_d = np.argmin( np.abs( np.cumsum(v_lam) - (1 - epsilon) ) ) + 1
        if trunc_d > d_max:
            trunc_d = d_max
        #Truncation
        u = u[:, :trunc_d]
        s = s[:trunc_d]
        b = b[:trunc_d, :]

        dimension_bond = s.shape[0]
        smat = np.zeros((dimension_bond, dimension_bond))
        np.fill_diagonal(smat, s)
        b = np.reshape(b, [dimension_bond, phys_dim, bond_dim_2], order = 'F')
        b = np.transpose(b, (0, 2, 1))
        u = u.dot(smat)
    else:
        print 'Unvalid argument to preprare_one_site()'
        sys.exit()

    return b, u, dimension_bond
