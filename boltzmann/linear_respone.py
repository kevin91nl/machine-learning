from numpy import linalg

def compute_stats(m, w):
    """
    m: 1-D array
    w: 2-D array
    """
    s_i = m
    A = -w
    for i in range(w.shape[0]):
        A[i][i] = 1 / (1 - m[i] * m[i])
    A_inv = linalg.inv(A)
    s_i_s_j = np.dot(m.reshape(len(m), 1), m.reshape(1, len(m))) + A_inv
    
    return s_i, s_i_s_j
