import numpy as np
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)

def mean_trace_float32(float[:,::1] A):
    cdef:
        int i,j
        float[:] t = np.zeros(A.shape[0]-1, dtype=np.float32)
    for i in range(0, A.shape[0]-1):
        for j in range(i+1, A.shape[0]):
            t[j-i-1] += A[i,j]

    for i in range(A.shape[0]-1):
        t[i] /= A.shape[0]-i-1

    return np.array(t)


@boundscheck(False)
@wraparound(False)

def mean_trace_float64(double[:,::1] A):
    cdef:
        int i,j
        double[:] t = np.zeros(A.shape[0]-1, dtype=np.float64)
    for i in range(0, A.shape[0]-1):
        for j in range(i+1, A.shape[0]):
            t[j-i-1] += A[i,j]

    for i in range(A.shape[0]-1):
        t[i] /= A.shape[0]-i-1

    return np.array(t)