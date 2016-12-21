import numpy as np

def hosvd(A, dst_dims):
    src_dims = A.shape
    num_dims = len(src_dims)
    Uks = []
    S = np.copy(A)
    for i in xrange(num_dims):
        if src_dims[i] == dst_dims[i]:
            Uk = np.identity(src_dims[i])
        else:
            axes = [j + 1 if j < i else 0 if j == i else j for j in xrange(num_dims)]
            Ak = np.reshape(np.transpose(A, axes), (src_dims[i], -1))
            print "reshaped from", src_dims, "to", Ak.shape
            U, s, V = np.linalg.svd(Ak)
            Uk = U[:,:dst_dims[i]]
        Uks.append(Uk)
        S = np.tensordot(S, Uk, axes=[[0],[0]])
        print S.shape
    return S, Uks

def conv_svd(A, R3, R4):
    S, Uks = hosvd(A, (R4, R3) + A.shape[2:])
    S = np.tensordot(S, Uks[2], [[2],[1]])
    S = np.tensordot(S, Uks[3], [[2],[1]])
    W1 = Uks[1].T[:,:,np.newaxis,np.newaxis]
    W2 = S
    W3 = Uks[0][:,:,np.newaxis,np.newaxis]
    """
    print A
    T = np.tensordot(W1, W2, [[0], [2]])
    T = np.tensordot(T, W3, [[3], [1]])
    T = np.transpose(T, [2, 0, 1, 3])
    print T
    """
    return W1, W2, W3

def ip_svd(A, C):
    S, Uks = hosvd(A, [C, C])
    W2 = Uks[0]
    W1 = np.tensordot(S, Uks[1], [[1], [1]])
    """
    print A
    print np.dot(W2, W1)
    """
    return W1, W2

#ip_svd(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]), 2)
#conv_svd(np.ones((2, 3, 4, 5)), 3, 4)

