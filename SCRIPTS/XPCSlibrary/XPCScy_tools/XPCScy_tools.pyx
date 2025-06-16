import numpy as np
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)

def mean_trace_float32(float[:,::1] A):
    cdef:
        int i,j
        double[:] t  = np.zeros(A.shape[0]-1, dtype=np.float64)
        double[:] t2 = np.zeros(A.shape[0]-1, dtype=np.float64)
        double N

    for i in range(0, A.shape[0]-1):
        for j in range(i+1, A.shape[0]):
            t[j-i-1]  += A[i,j]
            t2[j-i-1] += A[i,j]*A[i,j]

    for i in range(A.shape[0]-2):
        N = A.shape[0] - i - 1
        t2[i] =  np.sqrt((t2[i] - t[i]*t[i]/N) / (N*N))
        t[i]  /= N
    

    t2[A.shape[0]-2] = np.nan

    return np.array(t), np.array(t2)


@boundscheck(False)
@wraparound(False)

def mean_trace_float64(double[:,::1] A):
    cdef:
        int i,j
        double[:] t = np.zeros(A.shape[0]-1, dtype=np.float64)
        double[:] t2 = np.zeros(A.shape[0]-1, dtype=np.float64)
        double N

    for i in range(0, A.shape[0]-1):
        for j in range(i+1, A.shape[0]):
            t[j-i-1] += A[i,j]
            t2[j-i-1] += A[i,j]*A[i,j]

    for i in range(A.shape[0]-2):
        N = A.shape[0] - i - 1
        t2[i] =  np.sqrt((t2[i] - t[i]*t[i]/N) / (N*N))
        t[i]  /= N

    t2[A.shape[0]-2] = np.nan

    return np.array(t), np.array(t2)