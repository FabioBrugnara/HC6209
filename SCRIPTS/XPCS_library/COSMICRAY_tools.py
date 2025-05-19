"""
COSMICRAY_tools
===============

This module contains functions for filtering cosmic rays and gamma rays from E4M data.

Author: Fabio Brugnara
"""


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
        import PETRA3_tools as PETRA
        beamline = 'PETRA3'
        Nx, Ny, Npx, lxp, lyp = PETRA.Nx, PETRA.Ny, PETRA.Npx, PETRA.lxp, PETRA.lyp
    elif beamline_toset == 'ID10':
        import ID10_tools as ID10
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

def cosmic_filter(e4m_data, Dpx, counts_th,  mask=None, itime=None, Nfi=None, Nff=None, Lbin=None, mask_plot=False, hist_plot=False, Nsigma=10, MKL_library=True):
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
    Lbin: int, optional
        Binning factor. Default is None.
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
    if Lbin == None: Lbin = 1

    #  LOAD DATA
    t0 = time.time()
    print('Loading frames ...')
    if (Nfi!=0) or (Nff!=e4m_data.shape[0]): Itp = e4m_data[Nfi:Nff]
    else : Itp = e4m_data
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    # BIN DATA
    if Lbin != 1:
        t0 = time.time()
        if MKL_library:
            print('Binning frames (Lbin = '+str(Lbin)+', using MKL library) ...')
            Itp = (Itp[:Itp.shape[0]//Lbin*Lbin])
            BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0]), (np.arange(Itp.shape[0])//Lbin, np.arange(Itp.shape[0]))))
            Itp = dot_product_mkl(BIN_matrix, Itp, dense=False, cast=True)
            print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
    
        else:
            print('Binning frames (Lbin = '+str(Lbin)+') ...')
            Itp = Itp[:Itp.shape[0]//Lbin*Lbin]
            BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0]), (np.arange(Itp.shape[0])//Lbin, np.arange(Itp.shape[0]))))
            Itp = BIN_matrix@Itp
            print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    print('\t | '+str(Itp.shape[0])+' frames X '+str(Itp.shape[1])+' pixels')
    if isinstance(Itp, sparse.sparray):
        print('\t | sparsity = {:.2e}'.format(Itp.data.size/(Itp.shape[0]*Itp.shape[1])))
        print('\t | memory usage (sparse.csr_array @ '+str(Itp.dtype)+') =', round((Itp.data.nbytes+Itp.indices.nbytes+Itp.indptr.nbytes)/1024**3,3), 'GB')
    else:
        print('\t | memory usage (np.array @ '+str(Itp.dtype)+') =', round(Itp.nbytes/1024**3,3), 'GB')

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

    KM = sparse.block_array(KM, format='csr')
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    # KERNEL MULTIPLICATION (COSMIC RAY RETRIEVAL)
    t0 = time.time()
    if MKL_library:
        print('Cosmic ray retrieval (using MKL library) ...')
        CR = (dot_product_mkl(Itp.astype(bool), KM, cast=True) >= counts_th)*Itp.astype(bool)
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
    else:
        print('Cosmic ray retrieval ...')
        CR = ((Itp.astype(bool)@KM) >= counts_th)*Itp.astype(bool)
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    I_cosmic = (Itp*CR).sum()
    N_cosmic = CR.sum(axis=1).flatten().astype(bool).sum()

    if itime is not None:
        print('\t | Cosmic ray signal intensity =', I_cosmic/Itp.sum()*100, '% (', round(I_cosmic/(itime*Itp.shape[0]),0), 'counts/s)')
        print('\t | # of cosmic rays (assumption of max 1 event per frame) =', N_cosmic/Itp.shape[0]*100, '% of frames (', N_cosmic/(itime*Itp.shape[0]), 'events/s)')
    else:
        print('\t | Cosmic ray signal intensity =', I_cosmic/Itp.sum()*100, '%')
        print('\t | # of cosmic rays (assumption of max 1 event per frame) =', N_cosmic/Itp.shape[0]*100, '% of frames')

    # FIRST HIST PLOT
    if hist_plot:
        plt.figure(figsize=(10,5))
        It = Itp.sum(axis=1)
        It_mean = It.mean()
        It_std = np.sqrt((It**2).mean()-It_mean**2)
        plt.hist(It, alpha=1, range=(It_mean-Nsigma*It_std, It_mean+Nsigma*It_std), label='raw_data', bins=200)
        plt.yscale('log')
        plt.xlabel('Integrated intensity [counts/frame]')

    # REMOVE COSMIC RAYS
    t0 = time.time()
    print('Removing cosmic rays ...')
    Itp = Itp - (Itp*CR)
    if isinstance(Itp, sparse.sparray): Itp.eliminate_zeros()
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    # SECOND HIST PLOT
    if hist_plot:
        plt.hist(Itp.sum(axis=1), alpha=.5,  range=(It_mean-Nsigma*It_std, It_mean+Nsigma*It_std), label='cosmic ray removed', bins=200)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # MASK PLOT
    if mask_plot:
        plt.figure(figsize=(10,10))
        CR_mask = CR.sum(axis=0).astype(bool)
        plt.imshow(CR_mask.reshape(Nx, Ny), cmap='viridis', origin='lower')
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.title('Cosmic Ray Mask')
        plt.tight_layout()
        plt.show()

    return CR, Itp
