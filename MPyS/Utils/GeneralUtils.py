import numpy as np
import sys

# An SVD routine with a small change.
# Notice that unlike Matlab implementations there's no transposition of V at the end,
# given that numpy's SVD does this by default
def svd(t):
    try:
        U, S, V = np.linalg.svd(t, full_matrices = False)
    except np.linalg.linalg.LinAlgError:
        print "# Convergence problems occured in SVD!"
        U, S, V = np.linalg.svd(t + (1.0E-15 * np.random.randn(*t.shape)), full_matrices = False)

    return U, S, V

# A routine to evaluate expectation values.
# Arguments:
# mps is a one-dimensional Python list containing numpy arrays representing the MPS at each site
# hset is a 2D Python list containing numpy arrays representating the site operator for which the
# expectation value is to be calculated
def expectation_value(mps, hset, n, m):
    phys_dim = mps[0].shape[2]

    exp_val = 0.0
    local_val = np.zeros(m, dtype=complex)
    for nexps in range(m):
        exp_val_m = np.array([[1.0]])
        for sites in reversed(range(n)):
            if(m == 1):
                h = np.copy(hset[sites])
            else:
                h = np.copy(hset[nexps][sites])
            h = np.reshape(h, [1, 1, phys_dim, phys_dim], order = 'F')
            exp_val_m = update_c_right(exp_val_m, mps[sites], h, mps[sites])
        local_val[nexps] = exp_val_m[0][0]
        exp_val = exp_val + exp_val_m[0][0]

    # Norm
    norm = np.array([[1.0]])
    ident = np.identity(phys_dim)
    ident = np.reshape(ident, [1, 1, phys_dim, phys_dim], order = 'F')
    for sites in reversed(range(n)):
        norm = update_c_right(norm, mps[sites], ident, mps[sites])
    norm = norm[0][0]

    return local_val, exp_val / norm, norm

# A routine to evaluate expectation values for open systems.
# Arguments:
# mps is a one-dimensional Python list containing numpy arrays representing the MPS at each site
# hset is a 2D Python list containing numpy arrays representating the site operator for which the
# expectation value is to be calculated
def expectation_value_trace(mps, hset, n, m):
    phys_dim = mps[0].shape[2]

    mps_ident = np.identity(int(np.sqrt(phys_dim)))
    mps_ident = np.reshape(mps_ident, [1, 1, phys_dim])
    exp_val = 0.0
    local_val = np.zeros(m, dtype=complex)
    for nexps in range(m):
        exp_val_m = np.array([[1.0]])
        for sites in reversed(range(n)):
            if(m == 1):
                h = np.copy(hset[sites])
            else:
                h = np.copy(hset[nexps][sites])
            h = np.reshape(h, [1, 1, phys_dim, phys_dim], order = 'F')
            exp_val_m = update_c_right(exp_val_m, mps_ident, h, mps[sites])
        local_val[nexps] = exp_val_m[0][0]
        exp_val = exp_val + exp_val_m[0][0]

    # Norm
    norm = np.array([[1.0]])
    ident = np.identity(phys_dim)
    ident = np.reshape(ident, [1, 1, phys_dim, phys_dim], order = 'F')
    for sites in reversed(range(n)):
        norm = update_c_right(norm, mps_ident, ident, mps[sites])
    norm = norm[0][0]

    return local_val, exp_val / norm, norm

def update_c_right(c_right, tensor_b, tensor_x, tensor_a):
    if(tensor_x.shape[0] == 0):
        tensor_x = np.identity(tensor_b.shape[2])
        tensor_x = np.reshape(tensor_x, [1, 1, tensor_b.shape[2], tensor_b.shape[2]], order = 'F')

    c_right, numind_x = contract_tensors(tensor_a, 3, (1, ), c_right, 3, (2, ))
    c_right, numind_x = contract_tensors(tensor_x, 4, (1,3), c_right, 4, (3,1))
    c_right, numind_x = contract_tensors(tensor_b.conjugate(), 3, (1,2), c_right, 4, (3,1))

    return c_right

def update_c_left(c_left, tensor_b, tensor_x, tensor_a):
    if(tensor_x.shape[0] == 0):
        tensor_x = np.identity(tensor_b.shape[2])
        tensor_x = np.reshape(tensor_x, [1, 1, tensor_b.shape[2], tensor_b.shape[2]], order = 'F')

    c_left, numind_x = contract_tensors(tensor_a, 3, (0, ), c_left, 3, (2, ))
    c_left, numind_x = contract_tensors(tensor_x, 4, (0,3), c_left, 4, (3,1))
    c_left, numind_x = contract_tensors(tensor_b.conjugate(), 3, (0,2), c_left, 4, (3,1))

    return c_left

def contract_tensors(tensor_x, numind_x, ind_x, tensor_y, numind_y, ind_y):
    x_size = np.ones(numind_x, dtype = int)
    for i in range(len(tensor_x.shape)):
        x_size[i] = tensor_x.shape[i]
    y_size = np.ones(numind_y, dtype = int)
    for i in range(len(tensor_y.shape)):
        y_size[i] = tensor_y.shape[i]

    ind_x = np.asarray(ind_x)
    ind_y = np.asarray(ind_y)

    ind_x_l = [i for i in range(numind_x)]
    ind_x_l = np.array(ind_x_l)
    ind_y_r = [i for i in range(numind_y)]
    ind_y_r = np.array(ind_y_r)

    ind_x_l = np.delete(ind_x_l, ind_x, axis = 0)
    ind_y_r = np.delete(ind_y_r, ind_y, axis = 0)

    size_x_l = np.array( np.take(x_size, ind_x_l, axis = 0) )
    size_x   = np.array( np.take(x_size, ind_x, axis = 0) )

    size_y_r = np.array( np.take(y_size, ind_y_r, axis = 0) )
    size_y   = np.array( np.take(y_size, ind_y, axis = 0) )

    # Dimension check
    if(np.prod(size_x) != np.prod(size_y)):
        print 'ind_x and ind_y are not of the same dimension'
        sys.exit()

    if(ind_y_r.size == 0):
        if(ind_x_l.size == 0):
            tensor_x = np.transpose(tensor_x, ind_x)
            tensor_x = np.reshape(tensor_x, [1, np.prod(size_x)], order = 'F')

            tensor_y = np.transpose(tensor_y, ind_y)
            tensor_y = np.reshape(y, [np.prod(size_y), 1], order = 'F')

            tensor_x = tensor_x.dot(tensor_y)
        else:
            tensor_x = np.transpose(tensor_x, np.concatenate((ind_x_l, ind_x)))
            tensor_x = np.reshape(tensor_x, [np.prod(size_x_l), np.prod(size_x)], order = 'F')

            tensor_y = np.transpose(tensor_y, ind_y)
            tensor_y = np.reshape(tensor_y, [np.prod(size_y), 1], order = 'F')

            tensor_x = tensor_x.dot(tensor_y)

            xsize = np.take(x_size, ind_x_l, axis = 0)

            tensor_x = np.reshape(tensor_x, np.concatenate(xsize, [1]), order = 'F')
    else:
        while(len(tensor_x.shape) < ( len(ind_x) + len(ind_x_l) )):
            ax = len(tensor_x.shape)
            tensor_x = np.expand_dims(tensor_x, axis = ax)

        tensor_x = np.transpose(tensor_x, np.concatenate((ind_x_l, ind_x)))
        tensor_x = np.reshape(tensor_x, [np.prod(size_x_l), np.prod(size_x)], order = 'F')

        while(len(tensor_y.shape) < ( len(ind_y) + len(ind_y_r) )):
            ax = len(tensor_y.shape)
            tensor_y = np.expand_dims(tensor_y, axis = ax)

        tensor_y = np.transpose(tensor_y, np.concatenate((ind_y, ind_y_r)))
        tensor_y = np.reshape(tensor_y, [np.prod(size_y), np.prod(size_y_r)], order = 'F')

        tensor_x = tensor_x.dot(tensor_y)

        x_size = np.concatenate( (np.take(x_size, ind_x_l), np.take(y_size, ind_y_r)) )
        numind_x = len(x_size)

        if(x_size[-1] == 1):
            remove_indices = np.array([])
            for index in range(2, len(x_size)):
                if(x_size[index] == 1):
                    remove_indices = np.append(remove_indices, index)
            x_size = np.delete(x_size, remove_indices)

        tensor_x = np.reshape(tensor_x, x_size, order = 'F')
    return tensor_x, numind_x
