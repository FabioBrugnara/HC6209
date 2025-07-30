"""
COSMICRAY_tools
===============

This module contains functions for filtering cosmic rays and gamma rays from E4M data.

Author: Fabio Brugnara
"""

import os
os.environ["MKL_INTERFACE_LAYER"] = "ILP64"


### IMPORT SCIENTIFIC LIBRARIES ###
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy import sparse
from sparse_dot_mkl import dot_product_mkl


#############################################
##### SET BEAMLINE  AND EXP VARIABLES #######
#############################################

def set_beamline(beamline_toset):
    '''
    Set the beamline parameters for the data analysis.

    Parameters
    ----------
    beamline_toset: str
        Beamline name
    '''
    global beamline, Nx, Ny, Npx, lxp, lyp
    if beamline_toset == 'PETRA3':
        import PETRA3tools as PETRA
        beamline = 'PETRA3'
        Nx, Ny, Npx, lxp, lyp = PETRA.Nx, PETRA.Ny, PETRA.Npx, PETRA.lxp, PETRA.lyp
    elif beamline_toset == 'ID10':
        import ID10tools as ID10
        beamline = 'ID10'
        Nx, Ny, Npx, lxp, lyp = ID10.Nx, ID10.Ny, ID10.Npx, ID10.lxp, ID10.lyp
    else:
        raise ValueError('Beamline not recognized!')
    
############################################
########### GAMMA RAY FILTER ###############
############################################

def fast_gamma_filter(e4m_data, Imaxth_high, mask=None, info=False, itime=None):
    '''
    Fast gamma ray filter for E4M data.

    Notes
    -----
    Use mask=None and info=None drammatically improves the speed of the function.
    
    Parameters
    ----------
    e4m_data: sparse.sparray
        E4M data to be filtered.
    Imaxth_high: float
        Threshold for gamma ray signal.
    mask: sparse.sparray, optional
        Mask to be applied to the data. Default is None.
    info: bool, optional
        If True, print information about the gamma ray signal. Default is False.
    itime: float, optional
        Integration time in seconds. Default is None.
    
    Returns
    -------
    e4m_data: sparse.sparray
        Filtered E4M data.
    '''

    if mask is not None:
        t0 = time.time()
        print('Masking data (set 0s on ~mask pixels) ...')
        e4m_data = e4m_data*mask
        e4m_data.eliminate_zeros()
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    t0 = time.time()
    print('Filtering gamma ray signal (i.e. signals over treshold) ...')
    GR = e4m_data>Imaxth_high
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    if info:
        t0 = time.time()
        print('Computing informations ...')
        I_gamma = (e4m_data*GR).sum()
        N_gamma = GR.sum(axis=1).flatten().astype(bool).sum()
        if itime is not None:
            print('\t | Gamma ray signal intensity =', I_gamma / e4m_data.sum() * 100, '% (', round(I_gamma / (itime*e4m_data.shape[0]),0), 'counts/s)')
            print('\t | # of Gamma rays (assumption of max 1 gamma per frame) =', N_gamma/e4m_data.shape[0]*100, '% of frames (', N_gamma/(e4m_data.shape[0]*itime), 'ph/s)')
        else:
            print('\t | Gamma ray signal intensity =', I_gamma / e4m_data.sum() * 100, '%')
            print('\t | # of Gamma rays (assumption of max 1 gamma per frame) =', N_gamma/e4m_data.shape[0]*100, '% of frames')
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
    t0 = time.time()
    print('Removing gamma ray signal (set 0s) ...')
    e4m_data = e4m_data - e4m_data* GR
    e4m_data.eliminate_zeros()
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
    
    return e4m_data

    
################################# 
##### COSMIC RAY FILTER #########
#################################

def cosmic_filter(e4m_data, Dpx, counts_th,  mask=None, itime=None, Nfi=None, Nff=None):
    '''
    Cosmic ray filter for E4M data.
    
    Parameters
    ----------
    e4m_data: sparse.sparray
        E4M data to be filtered.
    Dpx: int
        Size of the kernel in pixels.
    counts_th: int
        Threshold for cosmic ray signal.
    mask: sparse.sparray, optional
        Mask to be applied to the data. Default is None.
    itime: float, optional
        Integration time in seconds. Default is None.
    Nfi: int, optional
        First frame to be loaded. Default is None.
    Nff: int, optional
        Last frame to be loaded. Default is None.
    mask_plot: bool, optional
        If True, plot the mask. Default is False.
    hist_plot: bool, optional
        If True, plot the histogram. Default is False.
    Nsigma: int, optional
        Number of standard deviations for the histogram. Default is 10.
    MKL_library: bool, optional
        If True, use MKL library for matrix multiplication. Default is True.
    Returns
    -------
    CR: sparse.sparray
        Cosmic ray mask.
    Itp: sparse.sparray
        Filtered E4M data.
    '''

    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]

    #  LOAD DATA
    t0 = time.time()
    print('Loading frames ...')
    if (Nfi!=0) or (Nff!=e4m_data.shape[0]): Itp = e4m_data[Nfi:Nff]
    else : Itp = e4m_data
    # convert to float32
    if Itp.dtype != np.float32:
        Itp = Itp.astype(np.float32)
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    #  MASK DATA
    t0 = time.time()
    if mask is not None:
        print('Masking data (set 0s on the mask) ...')
        Itp = (Itp*mask).tocsr()
        if isinstance(Itp, sparse.sparray): Itp.eliminate_zeros()
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    # GENERATE KERNEL MATRIX (Npx X Npx)
    t0 = time.time()
    print('Generating kernel matrix ...')
    offsets = [i for i in range(-Dpx, Dpx+1) if i!=0]
    #offsets = [i for i in range(-Dpx, Dpx+1)]
    IWY = sparse.diags_array([1]*len(offsets), offsets=offsets, shape=(Ny, Ny), format='csr')

    def Ns4KernelMatrix(N, Dpx, c):
        if (c>=Dpx) and (N-c-1>=Dpx):
            return c-Dpx if c-Dpx>=0 else 0, Dpx*2+1, N-c-1-Dpx if N-c-1-Dpx>=0 else 0
        elif (c<Dpx):
            return 0, Dpx+c+1, N-(Dpx+c+1)
        elif (N-c-1<Dpx):
            return N-(Dpx+N-c), Dpx+N-c, 0

    KM = [[]]*Nx
    for x in range(Nx):
        a, b, c = Ns4KernelMatrix(Nx, Dpx, x)
        KM[x] = a*[None] + b*[IWY] + c*[None]

    KM = sparse.block_array(KM, format='csr').astype(np.float32)
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    # KERNEL MULTIPLICATION (COSMIC RAY RETRIEVAL)
    t0 = time.time()
    print('Cosmic ray retrieval (using MKL library) ...')
    print('\t -> Kernel matrix multiplication ...')
    CR = dot_product_mkl(Itp.astype(bool).astype(np.float32), KM)
    print('\t -> Removing cosmic rays borders ...')
    CR = CR*Itp.astype(bool)
    print('\t -> Thresholding ...')
    CR = (CR >= counts_th)

    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    I_cosmic = (Itp*CR).sum()
    N_cosmic = CR.sum(axis=1).flatten().astype(bool).sum()

    if itime is not None:
        print('\t | Cosmic ray signal intensity =', I_cosmic/Itp.sum()*100, '% (', round(I_cosmic/(itime*Itp.shape[0]),0), 'counts/s)')
        print('\t | # of cosmic rays (assumption of max 1 event per frame) =', N_cosmic/Itp.shape[0]*100, '% of frames (', N_cosmic/(itime*Itp.shape[0]), 'events/s)')
    else:
        print('\t | Cosmic ray signal intensity =', I_cosmic/Itp.sum()*100, '%')
        print('\t | # of cosmic rays (assumption of max 1 event per frame) =', N_cosmic/Itp.shape[0]*100, '% of frames')

    # REMOVE COSMIC RAYS
    t0 = time.time()
    print('Removing cosmic rays ...')
    Itp = Itp - (Itp*CR)
    if isinstance(Itp, sparse.sparray): Itp.eliminate_zeros()
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    return CR, Itp
