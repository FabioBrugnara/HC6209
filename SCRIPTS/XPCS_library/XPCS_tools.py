"""
XPCS_tools
==========

A python library for XPCS data analysis. Use in combination with the library ID10_tools_ or PETRA3_tools to load the data.

Author: Fabio Brugnara
"""


### IMPORT LIBRARIES ###
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm                                        # for progress bar
from scipy import sparse                                     # for sparse array in scipy
from scipy.ndimage import gaussian_filter, gaussian_filter1d # for Gaussian filtering in G2t plotting
from sparse_dot_mkl import dot_product_mkl, gram_matrix_mkl  # for fast (dense and sparse) matrix multiplication
import numexpr as ne                                         # for parallelized numpy operations
from XPCScy_tools.XPCScy_tools import mean_trace_float64     # for C-implemented functions


### VARIABLES ###
of_value4plot = 2**32-1 # value for the overflow pixels in the plot
 
#############################################
##### SET BEAMLINE AND EXP VARIABLES #######
#############################################

def set_beamline(beamline_toset:str):
    '''
    Set the beamline parameters for the XPCS data analysis. The function load the correct varaibles (Nx, Ny, Npx, lxp, lyp) from the beamline tools.

    Parameters
    ----------
        beamline: str
            Beamline name ('PETRA3' or 'ID10')
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
    

def set_expvar(X0_toset:int, Y0_toset:int, L_toset:float):
    '''
    Set the experimental variables for the data analysis.

    Parameters
    ----------
        X0_toset: int
            X0 position of the beam center in pixels
        Y0_toset: int
            Y0 position of the beam center in pixels
        L_toset: float
            Distance from the sample to the detector in meters
    '''
    global X0, Y0, L
    X0, Y0, L = X0_toset, Y0_toset, L_toset

############################################
########## GENERALFUNCTIONS ################
############################################

def E2lambda(E):
    '''
    Convert X-ray energy in keV to wavelength in Angstroms.
    
    Parameters
    ----------
        E: float
            X-ray energy in keV
    Returns
    -------
    float
        Wavelength in Angstroms
    '''
    return 12.39842/E

def lambda2E(l):
    '''
    Convert X-ray wavelength in Angstroms to energy in keV.

    Parameters
    ----------
        l: float
            Wavelength in Angstroms
    Returns
    -------
    float
        X-ray energy in keV
    '''
    return 12.39842/l

def theta2Q(Ei, theta):
    '''
    Convert the scattering angle in degrees to Q in 1/A.

    Parameters
    ----------
        Ei: float
            Energy of the beam in keV
        theta: float
            Scattering angle in degrees
    Returns
    -------
    float
        Q value in 1/A
    '''
    return 4*np.pi*np.sin(np.deg2rad(theta)/2)/E2lambda(Ei)

def Q2theta(Ei, Q):
    '''
    Convert Q in 1/A to the scattering angle in degrees.

    Parameters
    ----------
        Ei: float
            Energy of the beam in keV
        Q: float
            Q value in 1/A
    Returns
    -------   
    float
        Scattering angle in degrees
    '''
    return 2*np.rad2deg(np.arcsin(E2lambda(Ei)*Q/4/np.pi))


def decorrelation_f(t, tau, beta, c, y0):
    '''
    Decorrelation function for XPCS analysis.

    Parameters
    ----------
        t : array_like
            Time variable.
        tau : float
            Characteristic decay time.
        beta : float
            Stretching exponent.
        c : float
            Contrast or amplitude.
        y0 : float
            Baseline offset.
    Returns
    -------
    array_like
        Decorrelation function evaluated at t.
    '''
    return c * np.exp(-(t / tau) ** beta) + y0

#########################################################################################################################


#################################
########### MASK PLOTS ##########
#################################

def gen_plots4mask(e4m_data, itime, Ith_high=None, Ith_low=None, Imaxth_high=None, mask=None, load_mask=None, mask_geom=None, Nfi=None, Nff=None, max_plots=False, wide_plots = False):
    '''
    Function that generates a number of different plots to create the mask! By default it generates the average flux per pixel map and histogram.
    
    Parameters
    ----------
        e4m_data: sparse.csr_matrix
            Sparse matrix of the e4m detector data
        itime: float
            Integration time of the e4m detector
        Ith_high: float
            Threshold (above) for the mean photon flux of the pixels [ph/s/px]
        Ith_low: float
            Threshold (below) for the mean photon flux of the pixels [ph/s/px]
        Imaxth_high: float
            Maximum number of counts per pixel treshold [ph/px]
        mask: np.array
            If e4m_data.shape[1]==Npx, is the mask to apply to the data for plotting and histogram generation.\n
            If e4m_data.shape[1]!=Npx, mask is assumed is the same used to load the data, thus mask.sum()==e4m_data.shape[1]. In this case mask is used to plot the correct X-Y profile.
        mask_geom: list of dicts
            List of geometries to mask. The function just plot the geometries on top of the XY profile.
        Nfi: int
            First frame to consider
        Nff: int
            Last frame to consider
        max_plots: bool
            If True, plot the maximum counts per pixel map and histogram.
        wide_plots: bool
            If True, plot the wide histogram of the mean flux per pixel and maximum counts per pixel (if max_plots is True).
    '''

    # CHECK e4m_data AND load_mask DIMENSION
    if (e4m_data.shape[1] != Npx) and (load_mask is None):     raise ValueError('Cannot generate plots from already masked data! Please provide the load_mask!')
    if (load_mask is not None) and (mask is not None):         raise ValueError('Cannot apply mask to masked loaded data!')
    if (e4m_data.shape[1] == Npx) and (load_mask is not None): raise ValueError('Cannot use load_mask with already masked data!')
    if load_mask is not None:
        if (e4m_data.shape[1] != load_mask.sum()):             raise ValueError('The load_mask does not match the e4m_data.shape[1]! Please check the mask dimensions!')
    
    
    # LOAD DATA in Nfi:Nff
    e4m_data = e4m_data[Nfi:Nff]

    # GENERATE MASK
    if mask is None: mask = np.ones(Npx, dtype=bool)

    # COMPUTE THE MEAN FLUX PER PX [ph/s/px]
    I_mean = np.ones(Npx)*of_value4plot
    if load_mask is None: I_mean[mask]      = e4m_data[:,mask].sum(axis=0)/(itime*e4m_data.shape[0])
    else:                 I_mean[load_mask] = e4m_data.sum(axis=0)        /(itime*e4m_data.shape[0])

    # COMPUTE THE MAXIMUM COUNTS PER PX [ph/px] (only if needed)
    if (Imaxth_high is not None) or max_plots:
        I_max = np.ones(Npx)*of_value4plot
        if e4m_data.shape[1] == Npx: I_max[mask] = np.array(e4m_data[mask].max(axis=0))
        else:                        I_max[mask] = np.array(e4m_data.max(axis=0))

    # PRINT INFORMATIONS    
    print('################################################################################')
    print('Maximum count in the hull run ->', e4m_data.max())
    if Ith_high is not None: print('# of pixels above Ith_high treshold -> ', I_mean[mask][I_mean[mask]>Ith_high].shape[0], 'pixels (of', I_mean.shape[0], '=>', round(I_mean[mask][I_mean[mask]>Ith_high].shape[0]/I_mean[mask].shape[0]*100, 2), '%)')
    if Ith_low is not None: print('# of pixels below Ith_low treshold -> ',   I_mean[mask][I_mean[mask]<Ith_low].shape[0], 'pixels (of', I_mean.shape[0], '=>', round(I_mean[mask][I_mean[mask]<Ith_low].shape[0]/I_mean[mask].shape[0]*100, 2), '%)')
    if Imaxth_high is not None: print('# of pixels above Imaxth_high treshold -> ', I_max[mask][I_max[mask]>Imaxth_high].shape[0], 'pixels (of', I_max.shape[0], '=>', round(I_max[mask][I_max[mask]>Imaxth_high].shape[0]/I_max[mask].shape[0]*100, 2), '%)')
    print('################################################################################\n')

    # MEAN FLUX PER PX FIGURE
    plt.figure(figsize=(8,13))
    ax4 = plt.subplot(211)
    im = ax4.imshow(I_mean.reshape(Nx, Ny), vmin=Ith_low, vmax=Ith_high, origin='lower')                                                                    # plot the mean flux per px 
    plt.colorbar(im, ax=ax4)                                                                                                                                # add colorbar, labels, ...  
    ax4.set_title('Mean flux per px [ph/s/px]')                                                                                             
    ax4.set_xlabel('Y [px]')
    ax4.set_ylabel('X [px]')
    ax4.plot(Y0, X0, 'ro', markersize=10)                                                                                                                   # plot the beam center
    if mask_geom is not None:                                                                                                                               # plot the mask geometry (mask_geom)
        for obj in mask_geom:                                                                                                                               # loop over the objects ...  
            if obj['geom'] == 'Circle':
                ax4.add_artist(plt.Circle((obj['Cy'], obj['Cx']), obj['r'], color='r', fill=False))
            elif obj['geom'] == 'Rectangle':
                ax4.add_artist(plt.Rectangle((obj['y0'], obj['x0']), obj['yl'], obj['xl'], color='r', fill=False))

    # MEAN FLUX PER PX HISTOGRAM (ZOOM)
    ax5 = plt.subplot(413)                                                                                                                                  # create the subplot
    if (Ith_high is not None) and (Ith_low is not None): ax5.hist(I_mean[mask], bins=200, range=(Ith_low*.5, Ith_high*1.5),       label='(zoom)')           # plot the histogram
    elif (Ith_high is not None) and (Ith_low is None):   ax5.hist(I_mean[mask], bins=200, range=(0, Ith_high*1.5),                label='(zoom)')           # ..
    elif (Ith_high is None) and (Ith_low is not None):   ax5.hist(I_mean[mask], bins=200, range=(Ith_low*.5, I_mean[mask].max()), label='(zoom)')           # ..
    else:                                                ax5.hist(I_mean[mask], bins=200,                                         label='(full range)')     # ..
    if Ith_high is not None: ax5.axvline(Ith_high, color='r', label='Ith_high')                                                                             # plot the Ith_high limit
    if Ith_low is not None:  ax5.axvline(Ith_low,  color='g', label='Ith_low')                                                                              # plot the Ith_low limit
    ax5.set_yscale('log')                                                                                                                                   # add the labels and legend
    ax5.set_xlabel('Mean flux per px [ph/s/px]')                                                                                                            # ..   
    ax5.legend()                                                                                                                                            # ..        

    # MEAN FLUX PER PX HISTOGRAM (FULL RANGE)
    if wide_plots:
        ax6 = plt.subplot(414)                                                                                                                              # create the subplot
        ax6.hist(I_mean[mask], bins=200, label='(full range)')                                                                                              # plot the histogram    
        if Ith_high is not None: ax6.axvline(Ith_high, color='r', label='Ith_high')
        if Ith_low  is not None: ax6.axvline(Ith_low,  color='g', label='Ith_low')
        ax6.set_yscale('log')
        ax6.set_xlabel('Mean flux per px[ph/s/px]')
        ax6.legend()

    plt.tight_layout(); plt.show()

    # MAXIMUM COUNTS PER PX FIGURE
    if max_plots:
        plt.figure(figsize=(8,13))
        ax4 = plt.subplot(211)

        # MAX COUNTS PER PX IMAGE
        im = ax4.imshow(I_max.reshape(Nx, Ny), vmin=0, vmax=Imaxth_high, origin='lower')
        plt.colorbar(im, ax=ax4)
        ax4.set_title('Max counts per px [ph/px]')
        ax4.set_xlabel('Y [px]')
        ax4.set_ylabel('X [px]')

        # MAX COUNTS PER PX HISTOGRAM (ZOOM)
        ax5 = plt.subplot(413)
        if Imaxth_high is not None: 
            ax5.hist(I_max[mask], bins=100, label='(zoom)', range=(0, Imaxth_high*1.5))
            ax5.axvline(Imaxth_high, color='r', label='Imaxth_high')
        else:
            ax5.hist(I_max[mask], bins=100, label='(full range)')

        # add labels and legend
        ax5.set_yscale('log')
        ax5.set_xlabel('Max counts per px [ph/px]')
        ax5.legend()

        # MAX COUNTS PER PX HISTOGRAM (FULL RANGE)
        if wide_plots:
            ax6 = plt.subplot(414)
            ax6.hist(I_max[mask], bins=200, label='(full range)')
            ax6.set_yscale('log')
            ax6.set_xlabel('Max counts per px [ph/px]')
            ax6.legend()

        plt.tight_layout()
        plt.show()

    
#################################
########### MASK GEN ############
#################################

def gen_mask(e4m_data=None, itime=None, mask=None, mask_geom=None, Ith_high=None, Ith_low=None, Imaxth_high=None, Nfi=None, Nff=None, hist_plots=False):
    '''
    Generate a mask for the e4m detector from various options. The function plot the so-obtained mask, and also return some histograms to look at the results (if hist_plots is True).

    Parameters
    ----------
    e4m_data: sparse.csc_matrix
        Sparse matrix of the e4m detector data
    itime: float
        Integration time of the e4m detector
    mask: np.array
        Mask of the e4m detector lines (slightly wider than the overflow lines, as pixels on the adges are not reliable)
    mask_geom: list of dicts
        List of geometries to mask (in dictionary form). The supported objects are:\n
        - Circle: {'geom': 'Circle', 'Cx': x0, 'Cy': y0, 'r': r, 'inside': True/False}\n
        - Rectangle: {'geom': 'Rectangle', 'x0': x0, 'y0': y0, 'xl': xl, 'yl': yl, 'inside': True/False}\n
        Example:\n
        mask_geom = [   {'geom': 'Circle', 'Cx': 100, 'Cy': 100, 'r': 10, 'inside': True}, {'geom': 'Rectangle', 'x0': 50, 'y0': 50, 'xl': 20, 'yl': 10, 'inside': False}]
    Ith_high: float
        Threshold (above) for the mean photon flux of the pixels
    Ith_low: float
        Threshold (below) for the mean photon flux of the pixels
    Imaxth_high: float
        Maximum number of counts per pixel treshold
    Nfi: int
        First frame to consider
    Nff: int
        Last frame to consider  
    hist_plots: bool
        If True, plot the histograms of the mean flux per pixel and maximum counts per pixel.

    Returns
    -------
    np.array
        Mask of the e4m detector
    '''

    # CHECK e4m_data DIMENSION & LOAD DATA in Nfi:Nff
    if e4m_data is not None: 
        if e4m_data.shape[1] != Npx: raise ValueError('Cannot generate a mask from already masked data!')
        e4m_data = e4m_data[Nfi:Nff]

    # GENERATE MASK OF ONES (if mask is None)
    if mask is None: mask = np.ones(Npx, dtype=bool)

    # APPLAY GEOMETRIC MASKS (if mask_geom is not None)
    if (mask_geom is not None) and (mask_geom!=[]):
        mask = mask.reshape(Nx, Ny)
        X, Y = np.mgrid[:Nx, :Ny]
        for obj in mask_geom:
            if obj['geom']=='Circle':
                if obj['inside']:
                    mask = mask * ((Y-obj['Cy'])**2 + (X-obj['Cx'])**2 <= obj['r']**2)
                else:
                    mask = mask * ((Y-obj['Cy'])**2 + (X-obj['Cx'])**2 > obj['r']**2)
            elif obj['geom']=='Rectangle':
                if obj['inside']:
                    mask = mask * ((Y>obj['y0']) & (Y<obj['y0']+obj['yl']) & (X>obj['x0']) & (X<obj['x0']+obj['xl']))
                else:
                    mask = mask * ((Y<obj['y0']) | (Y>obj['y0']+obj['yl']) | (X<obj['x0']) | (X>obj['x0']+obj['xl']))
        mask = mask.flatten()

    # FILTER USING THRESHOLDS (Ith_high, Ith_low, Imaxth_high) & AND COMPUTING I_mean, I_max (if needed)
    if (Ith_high is not None) or (Ith_low is not None) or (hist_plots==True):
        I_mean = e4m_data.sum(axis=0)/(itime*e4m_data.shape[0])
        if Ith_high is not None: mask = mask * (I_mean<=Ith_high)
        if Ith_low is not None : mask = mask * (I_mean>=Ith_low)
    if (Imaxth_high!=None) or (hist_plots==True):
        I_max = np.array(e4m_data.max(axis=0))
        if Imaxth_high is not None : mask = mask * (I_max<=Imaxth_high)

    # PRINT PERCENTAGE OF MASKED PIXELS
    print('#################################################')
    print('Masked area = ', mask.sum()/Npx*100, '%')
    print('#################################################\n')

    # PLOT THE MASK
    plt.figure(figsize=(8,8))
    plt.imshow(mask.reshape((Nx, Ny)), origin='lower')
    plt.xlabel('Y [px]')
    plt.ylabel('X [px]')
    plt.tight_layout()
    plt.show()

    # PLOT THE HISTOGRAMS (if hist_plots is True)
    if hist_plots==True:
        plt.figure(figsize=(8,6))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        # Masked histogram of px flux
        ax1.set_title('Masked histogram of px flux')
        ax1.hist(I_mean[mask], bins=100)
        ax1.set_yscale('log')
        ax1.set_xlabel('Mean flux per px')

        # Maked histogram of max counts per px
        ax2.set_title('Masked histogram of max counts per px')
        ax2.hist(I_max[mask].data, bins=30, label='no zero counts')
        ax2.legend()
        ax2.set_yscale('log')
        ax2.set_xlabel('Max counts per px')
        plt.tight_layout()
        plt.show()

    return mask


#################################
########## Q MASK GEN ###########
#################################

def gen_Qmask(Ei, theta, Q, dq, Qmap_plot=False):
    '''
    Generate the Q masks for the given Q values at the working angle. The function also plot the Qmap for the given energy and angle (if Qmap_plot is True).

    Parameters
    ----------
    Ei: float
        Energy of the beam in keV
    theta: float
        Working angle in degrees
    Q: float or list of floats
        Q value(s) to mask in [1/A]
    dq: float or list of floats
        Q width(s) to mask in [1/A]
    Qmap_plot: bool
        If True, plot the Qmap for the given energy and angle

    Returns
    -------
    np.array or dict of np.array
        Q mask(s) of the e4m detector
    '''

    # GET THE X-Y MAPS
    X, Y = np.mgrid[:Nx, :Ny]

    # COMPUTE THE Q MAP FOR THE GIVEN DETECTOR DISTANCE, DETECTOR POSITION, AND X-RAY ENERGY
    if beamline=='ID10':
        dY0 = L*np.tan(np.deg2rad(theta))
        dY0_map =np.sqrt(((X-X0)*lxp)**2+(dY0-(Y-Y0)*lyp)**2)
        theta_map = np.arctan(dY0_map/L)
    elif beamline=='PETRA3':
        dX0 = L*np.tan(np.deg2rad(theta))
        dX0_map =np.sqrt(((dX0-(X-X0)*lxp))**2+((Y-Y0)*lyp)**2)
        theta_map = np.arctan(dX0_map/L)
    Q_map = theta2Q(Ei, np.rad2deg(theta_map))

    # GET THE Q REGION
    if (type(Q) == float) or (type(Q) == int):           Qmask       = (np.abs(Q_map-Q)<dq      ).flatten() # case of a single Q value
    else:
        Qmask = {}
        for i in range(len(Q)):
            if (type(dq) == float) or (type(dq) == int): Qmask[Q[i]] = (np.abs(Q_map-Q[i])<dq   ).flatten() # case of a list of Q values, single dq value
            else:                                        Qmask[Q[i]] = (np.abs(Q_map-Q[i])<dq[i]).flatten() # case of a list of Q values, list of dq values
    
    # QMASK PLOT
    plt.figure(figsize=(8,8))                                          
    if (type(Q) == float) or (type(Q) == int):                                                              # case of a single Q value
        plt.imshow(Qmask.reshape((Nx, Ny)), cmap='viridis', origin='lower', vmin=0, vmax=1, alpha=1)
        plt.scatter([],[], color=plt.cm.viridis(1.), label=str(Q)+'$\\AA^{-1}$')
    else:                                                                                                   # case of a list of Q values
        Qmask2plot = 0
        s = 1/len(Q)
        for i, q in enumerate(Qmask.keys()):
            Qmask2plot += Qmask[q].reshape((Nx, Ny))*s*(i+1)
            plt.scatter([],[], color=plt.cm.viridis(s*(i+1)), label=str(Q[i])+'$\\AA^{-1}$')
        plt.imshow(Qmask2plot, cmap='viridis', origin='lower', vmin=0, vmax=1, alpha=1)

    plt.xlabel('Y [px]'); plt.ylabel('X [px]'); plt.legend()
    plt.tight_layout(); plt.show()

    # QMAP PLOT (if Qmap_plot=True)
    if Qmap_plot:
        plt.figure(figsize=(8,8))
        plt.imshow(Q_map, cmap='viridis', origin='lower')
        plt.colorbar(); plt.xlabel('Y [px]'); plt.ylabel('X [px]'); plt.title('Q [$\\AA^{-1}$]')
        plt.tight_layout(); plt.show()

    return Qmask


############################
######## GET It ############
############################

def get_It(e4m_data, itime, mask=None, Nfi=None, Nff=None, Lbin=None, Nstep=None):
    '''
    Compute the average frame intensity [ph/px/s] vector from the e4m_data, properly masked with the mask. 
    
    Parameters
    ----------
    e4m_data: sparse.csr_matrix
        Sparse matrix of the e4m detector data
    itime: float
        Integration time of the e4m detector
    mask: np.array
        Mask of the e4m detector
    Nfi: int
        First frame to consider
    Nff: int
        Last frame to consider
    Lbin: int
        Binning factor for the frames
    Nstep: int
        Step for the frames

    Returns
    -------
    t_It: np.array
        Time array for the It vector
    It: np.array
        It vector
    '''
    # DEFAULT VALUES
    if Nfi is None: Nfi = 0
    if Nff is None: Nff = e4m_data.shape[0]
    if Lbin is None: Lbin = 1
    if Nstep is None: Nstep = 1
    if mask is None: mask = np.ones(e4m_data.shape[1], dtype=bool) 
    
    # COMPUTE It (masked)
    idx = Nfi + np.array([i for i in range(Nff-Nfi) if i % Nstep < Lbin][:((Nff-Nfi)//Nstep-1)*Nstep]) # GET THE CORRECT INDEXES FROM Nfi, Nff, Lbin and Nstep
    It = e4m_data[idx][:,mask].sum(axis=1)/mask.sum()                                                  # Compute It
    if Lbin != 1: It = It[:(It.size//Lbin)*Lbin].reshape(-1, Lbin).sum(axis=1) / Lbin                  # BIN It (if Lbin > 1)
    It /= itime                                                                                        # NORMALIZE It
    t_It = np.linspace(Nfi*itime, Nff*itime, It.shape[0])                                              # BUILD THE TIME VECTOR    

    return t_It, It


#################################
######### COMUPTE G2t ###########
#################################

def get_G2t(e4m_data, mask=None, Nfi=None, Nff=None, Lbin=None):
    '''
    Compute the G2t matrix from the e4m, properly masked with the matrix mask.

    Parameters
    ----------
    e4m_data: sparse.csr_matrix
        Sparse matrix of the e4m detector data
    mask: np.array
        Mask of the e4m detector
    Nfi: int
        First frame to consider
    Nff: int
        Last frame to consider
    Lbin: int
        Binning factor for the frames
    MKL_library: boolean
        If True, use the MKL library for the matrix multiplication
    
    Returns
    -------
    G2t: np.array
        G2t matrix
        
    '''

    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]
    if Lbin == None: Lbin = 1

    #  LOAD DATA
    t0 = time.time()
    print('Loading frames ...')
    if (Nfi!=0) or (Nff!=e4m_data.shape[0]): Itp = e4m_data[Nfi:Nff]
    else : Itp = e4m_data
    # convert to float32
    if Itp.dtype != np.float32:
        Itp = Itp.astype(np.float32)
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    # BIN DATA
    if Lbin != 1:
        t0 = time.time()
        print('Binning frames (Lbin = '+str(Lbin)+', using MKL library) ...')
        Itp = (Itp[:Itp.shape[0]//Lbin*Lbin]) # throw the last frames 
        BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0]), (np.arange(Itp.shape[0])//Lbin, np.arange(Itp.shape[0]))), dtype=np.float32)
        Itp = dot_product_mkl(BIN_matrix, Itp)
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
        print('Masking data ...')
        Itp = Itp[:,mask]
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
        print('\t | '+str(Itp.shape[0])+' frames X '+str(Itp.shape[1])+' pixels')
        if isinstance(Itp, sparse.sparray):
            print('\t | sparsity = {:.2e}'.format(Itp.data.size/(Itp.shape[0]*Itp.shape[1])))
            print('\t | memory usage (sparse.csr_array @ '+str(Itp.dtype)+') =', round((Itp.data.nbytes+Itp.indices.nbytes+Itp.indptr.nbytes)/1024**3,3), 'GB')
        else:
            print('\t | memory usage (np.array @ '+str(Itp.dtype)+') =', round(Itp.nbytes/1024**3,3), 'GB')
    
    # Compute G2t
    t0 = time.time()
    print('Computing G2t (using MKL library)...')
    G2t = gram_matrix_mkl(Itp, dense=True, transpose=True)
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
    print('\t | '+str(G2t.shape[0])+' X '+str(G2t.shape[1])+' squared matrix')
    print('\t | memory usage (np.array @ '+str(G2t.dtype)+') =', round(G2t.nbytes/1024**3,3), 'GB')
           
    # Normalize G2t
    t0 = time.time()
    print('Normalizing G2t (using NumExpr library)...')
    It = Itp.sum(axis=1, dtype=np.float32)
    np.divide(np.sqrt(Itp.shape[1]), It, where=It>0, out=It, dtype=np.float32)
    Itr = It[:, None] # q[:, None] -> q.reshape(N, 1)
    Itc = It[None, :] # q[None, :] -> q.reshape(1, N)
    ne.evaluate('G2t*Itr*Itc', out=G2t)
    
    # Remove diagonal and fill no counts frames
    G2t[G2t.diagonal()==0, :] = 1
    G2t[:, G2t.diagonal()==0] = 1
    np.fill_diagonal(G2t, 0)
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)\n')
    return G2t



##########################################
######### COMUPTE G2t bunnched ###########
##########################################

def get_G2t_bybunch(e4m_data, Nbunch, mask=None, Nfi=None, Nff=None, Lbin=None):
    '''
    Compute the G2t matrix from the e4m, bunching the frames in Nbunch bunches, thus averaging the G2t matrix over the bunches. 

    Parameters
    ----------
    e4m_data: sparse.csc_matrix
        Sparse matrix of the e4m detector data
    Nbunch: int
        Number of bunches to consider
    mask: np.array
        Mask of the e4m detector
    Nfi: int
        First frame to consider
    Nff: int
        Last frame to consider
    Lbin: int
        Binning factor for the frames

    Returns
    -------
    G2t: np.array
        G2t matrix
    '''

    # DEFAULT VALUES FOR Nfi, Nff, Lbin
    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]
    if Lbin == None: Lbin = 1

    # GET BUNCHES LENGHT [fms]
    Lbunch = (Nff-Nfi)//Nbunch

    # PREPARE THE G2t MATRIX
    G2t = np.zeros((Lbunch//Lbin, Lbunch//Lbin), dtype=np.float64)
    
    # COMPUTE G2t FOR EACH BUNCH
    for i in range(Nbunch):
        print('Computing G2t for bunch', i+1, '(Nfi =', Nfi+i*Lbunch, ', Nff =', Nfi+(i+1)*Lbunch, ') ...')
        G2t += get_G2t(e4m_data, mask, Nfi=Nfi+i*Lbunch, Nff=Nfi+(i+1)*Lbunch, Lbin=Lbin)
        print('Done!\n')

    return G2t/Nbunch



##############################
######### GET g2 #############
##############################

def get_g2(dt, G2t, cython=False):
    '''
    Compute the g2 from the G2t matrix.

    Parameters
    ----------
    dt: float
        Time step between frames
    G2t: np.array
        G2t matrix
    cython: boolean
        If True, use the cython code to compute the g2

    Returns
    -------
    t: np.array
        Time array
    g2: np.array
        g2 array
    '''
    
    t0 = time.time()
    if cython:
        print('Computing g2 (using cython code)...')
        g2 = mean_trace_float64(G2t)
    else:
        print('Computing g2...')
        g2 = np.array([G2t.diagonal(i).mean() for i in range(1,G2t.shape[0])])
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)\n')

    return np.arange(1, len(g2)+1)*dt, g2



##############################
######### GET g2 multitau ####
##############################

def get_g2mt_fromling2(dt, g2):
    '''
    Compute the multitau g2 from the g2 array.

    Parameters
    ----------
    dt: float
        Time step between frames
    g2: np.array
        g2 array    

    Returns
    -------
    t_multit: np.array
        Time array for the multitau g2
    g2_multit: np.array
        Multitau g2 array
    '''

    t = (np.arange(len(g2))+1)*dt

    g2_multit = []
    t_multit = []
    for i in range(int(np.log2(len(g2)))):
        g2_multit.append(g2[2**i-1:2**(i+1)-1].mean())
        t_multit.append (t [2**i-1:2**(i+1)-1].mean())

    i +=1
    g2_multit.append(g2[2**i-1:].mean())
    t_multit.append(t[2**i-1:].mean())

    return t_multit, g2_multit

def get_g2_mt(dt, g2):
    '''
    Alias for backwords compatibility of XPCS.get_g2mt_fromling2 .
    Compute the multitau g2 from the g2 array.
    '''
    print('WARNING: get_g2_mt is deprecated. Use get_g2mt_fromling2 instead.')
    return get_g2mt_fromling2(dt, g2)


###########################
##### PLOT X/Y PROFILE ####
###########################

# def plot_XYprofile(e4m_data, itime, ax='Y', mask=None, Nfi=None, Nff=None):
#     '''
#     Plot the X or Y profiles of the e4m detector.

#     Parameters
#     ----------
#     e4m_data: sparse.csc_matrix
#         Sparse matrix of the e4m detector data
#     itime: float
#         Integration time of the e4m detector
#     ax: str
#         Axis to plot ('X' or 'Y')
#     mask: np.array
#         Mask of the e4m detector
#     Nfi: int
#         First frame to consider
#     Nff: int
#         Last frame to consider
#     '''

#     # DEFAULT VALUES FOR Nfi, Nff
#     if Nfi == None: Nfi = 0
#     if Nff == None: Nff = e4m_data.shape[0]

#     # COMPUTE It
#     It = (e4m_data[Nfi:Nff].sum(axis=0)/((Nff-Nfi)*itime)).reshape(Nx, Ny)

#     # APPLY MASK
#     if mask is not None: It[~mask.reshape(Nx, Ny)] = 0

#     # PLOT
#     plt.figure(figsize=(8,5))
#     if (ax=='Y') or (ax=='y'):
#         plt.plot(It.sum(axis=0)/mask.reshape(Nx, Ny).sum(axis=0))
#         plt.xlabel('Y [px]')
#     if (ax=='X') or (ax=='x'):
#         plt.plot(It.sum(axis=1)/mask.reshape(Nx, Ny).sum(axis=1))
#         plt.xlabel('X [px]')
#     plt.ylabel('Mean flux per px [ph/s/px]')
    
#     plt.tight_layout()
#     plt.show()




######################
###### PLOT G2T ######
######################

def plot_G2t(G2t, vmin, vmax, itime=None, t1=None, t2=None, x1=None, x2=None, sigma_filter=None, full=False):
    ''''
    Plot the G2t matrix.

    Parameters
    ----------
    G2t: np.array
        G2t matrix
    vmin: float
        Minimum value for the color scale
    vmax: float
        Maximum value for the color scale
    itime: float
        Integration time of the e4m detector
    t1: float
        First time to consider (in [s] if itime is provided, otherwise in [frames])
    t2: float
        Last time to consider (in [s] if itime is provided, otherwise in [frames])
    x1: float
        If provided, shift the x axis to the given initial value
    x2: float
        If provided, shift the x axis to the given final value
    sigma_filter: float
        Sigma for the Gaussian filter (in [frames]) 
    full: boolean
        If True, plot the full G2t matrix mirroring the lower part
    '''

    # BRHAVIORS WHEN t1, t2 ARE NONE
    if t1 is None: t1 = 0
    if (t2 is None) and (itime is None): t2 = G2t.shape[0]
    elif (t2 is None) and (itime is not None): t2 = G2t.shape[0]*itime

    # BEHAVIOURS WHEN t2 IS BIGGER THAN THE G2t MATRIX
    if (itime is None) and (t2>G2t.shape[0]): t2 = G2t.shape[0]
    elif (itime is not None) and (t2>G2t.shape[0]*itime): t2 = G2t.shape[0]*itime

    # BEHAVIOURS WHEN x1, x2 ARE NONE
    if (x1 is None) and (x2 is None): x1, x2 = t1, t2
    elif x1 is None: x1 = 0
    elif x2 is None: x2 = G2t.shape[1]

    # CUT THE G2t MATRIX
    if itime is None: G2t = G2t[t1:t2, x1:x2]
    else: G2t = G2t[int(t1//itime):int(t2//itime), int(x1//itime):int(x2//itime)]

    # APPLY GAUSSIAN FILTER (if sigma_filter is not None)
    if sigma_filter is not None:
        # default values for the filter
        truncate = 4
        radius = 2*round(truncate*sigma_filter) + 1 + truncate

        # enlarge the matrix above the diagonal
        for i in range(1, int(radius)+1):
            idx = range(i, G2t.shape[0]), range(G2t.shape[0]-i)
            G2t[idx] = G2t.diagonal(offset=i)

        # apply the filter
        G2t = gaussian_filter(G2t, sigma=sigma_filter, mode='nearest', truncate=4)

        # remove the enlarged part
        for i in range(1, int(radius)*4+1):
            idx = range(i, G2t.shape[0]), range(G2t.shape[0]-i)
            G2t[idx] = 0

    # ADD THE MIRRORING (if full==True)
    if full==True:
        G2t += G2t.T


    # PLOT
    plt.figure(figsize=(8,8))
    plt.imshow(G2t, vmin=vmin, vmax=vmax, origin='lower')

    # add ticks
    plt.yticks(np.round(np.linspace(0, G2t.shape[0], 6)).astype(int), np.round(np.linspace(x1, x2, 6)).astype(int))
    plt.xticks(np.round(np.linspace(0, G2t.shape[1], 6)).astype(int), np.round(np.linspace(t1, t2, 6)).astype(int))

    # add labels and colorbar
    plt.xlabel('Time [s]')
    plt.ylabel('Time [s]')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


##################################################### MULTI-TAU #####################################################

def _get_symG2t(Itp):
    # Compute G2t upper triangle
    G2t = gram_matrix_mkl(Itp, dense=True, transpose=True)

    # Normalize G2t (directly accounting for 0-counts frames)
    It = Itp.sum(axis=1, dtype=np.float32)
    It = np.where(It>0, It, np.sqrt(Itp.shape[1], dtype=np.float32))
    np.divide(np.sqrt(Itp.shape[1]), It, where=It>0, out=It, dtype=np.float32)
    Itr, Itc = It[:, None], It[None, :]
    ne.evaluate('G2t*Itr*Itc', out=G2t)
    return G2t

def _get_nonsymG2t(Itp1, Itp2):
    # Compute full G2t
    G2t = dot_product_mkl(Itp1, Itp2.T, dense=True)
           
    # Normalize G2t (directly accounting for 0-counts frames)
    It1 = Itp1.sum(axis=1, dtype=np.float32)
    It2 = Itp2.sum(axis=1, dtype=np.float32)
    It1 = np.where(It1>0, It1, np.sqrt(Itp1.shape[1], dtype=np.float32))
    It2 = np.where(It2>0, It2, np.sqrt(Itp2.shape[1], dtype=np.float32))
    np.divide(np.sqrt(Itp1.shape[1]), It1, out=It1, dtype=np.float32)
    np.divide(np.sqrt(Itp2.shape[1]), It2, out=It2, dtype=np.float32)
    Itr, Itc = It1[:, None], It2[None, :]
    ne.evaluate('G2t*Itr*Itc', out=G2t)
    return G2t

def _G2t2G2tmt(G2t, G2tmt, type):
    for b in range(int(np.log2(G2t.shape[0]))):
        if type=='sym':
            G2tmt[b].append(G2t.diagonal(offset=1))
        elif type=='non-sym':
            G2tmt[b].append(np.array([G2t[-1,0]]))

        BIN_matrix = sparse.csr_array((np.ones(G2t.shape[0]), (np.arange(G2t.shape[0])//2, np.arange(G2t.shape[0]))), dtype=np.float32)
        G2t = dot_product_mkl(BIN_matrix, G2t)
        G2t = dot_product_mkl(BIN_matrix, G2t.T)
        G2t= G2t.T/4
    return G2tmt


#########################################
##### GET multitau G2t 4 dense data #####
#########################################

def get_G2tmt_4dense(e4m_data, dense_depth, mask=None, Nfi=None, Nff=None):

    print('WARNING: the function is still not tested!')

    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]

    #  LOAD DATA
    t0 = time.time()
    print('Loading frames ...')
    if (Nfi!=0) or (Nff!=e4m_data.shape[0]): Itp = e4m_data[Nfi:Nff]
    else : Itp = e4m_data
    if Itp.dtype != np.float32: Itp = Itp.astype(np.float32) # CONVERT TO float32

    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    #  MASK DATA
    t0 = time.time()
    if mask is not None:
        print('Masking data ...')
        Itp = Itp[:,mask]
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')
        print('\t | '+str(Itp.shape[0])+' frames X '+str(Itp.shape[1])+' pixels')
        print('\t | memory usage (np.array @ '+str(Itp.dtype)+') =', round(Itp.nbytes/1024**3,3), 'GB')

    ### CHECK PARAMS CONDIOTIONS ###
    if Itp.shape[0]//2**dense_depth != Itp.shape[0]/2**dense_depth: raise ValueError('# of frames must be a multiple of 2^dense_depth!')

    ### DENSE COMPUTATION ###

    t0 = time.time()
    print('Computing dense multitau G2t...')
    G2tmt = []
    # recurevly compute G2t first diagonal and bin by a factor 2
    for i in tqdm(range(dense_depth)):
        G2t = (Itp[:-1] * Itp[1:]).sum(axis=1)  # G2t = <Itp*Itp(t-shifted)>p
        norm = np.sqrt(Itp.shape[1])/Itp.sum(axis=1) # standard normalization
        G2t = G2t * norm[1:] * norm[:-1]
        G2tmt.append(np.array(G2t))

        # bin Itp by a factor 2
        BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0]), (np.arange(Itp.shape[0])//2, np.arange(Itp.shape[0]))), dtype=np.float32)
        Itp = dot_product_mkl(BIN_matrix, Itp, cast=True)
        
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    return G2tmt


##########################################
##### GET multitau G2t 4 sparse data #####
##########################################

def get_G2tmt_4sparse(e4m_data, sparse_depth, dense_depth, mask=None, Nfi=None, Nff=None):

    # DEFAULT VALUES
    if Nfi == None: Nfi = 0
    if Nff == None: Nff = e4m_data.shape[0]

    #  LOAD DATA
    t0 = time.time()
    print('Loading frames ...')
    if (Nfi!=0) or (Nff!=e4m_data.shape[0]): Itp = e4m_data[Nfi:Nff]
    else : Itp = e4m_data
    if Itp.dtype != np.float32:
        Itp = Itp.astype(np.float32) # convert to float32
        print('Converting to float32 ...')
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    #  MASK DATA
    t0 = time.time()
    if mask is not None:
        print('Masking data ...')
        Itp = Itp[:,mask]
        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    # PRINT DATA INFO
    print('\t | '+str(Itp.shape[0])+' frames X '+str(Itp.shape[1])+' pixels')
    print('\t | sparsity = {:.2e}'.format(Itp.data.size/(Itp.shape[0]*Itp.shape[1])))
    print('\t | memory usage (sparse.csr_array @ '+str(Itp.dtype)+') =', round((Itp.data.nbytes+Itp.indices.nbytes+Itp.indptr.nbytes)/1024**3,3), 'GB')

    ### CHECK PARAMS CONDIOTIONS ###
    if sparse_depth >= dense_depth: raise ValueError('sparse_depth must be less/equal than dense_depth!')
    if Itp.shape[0]//2**dense_depth != Itp.shape[0]/2**dense_depth: raise ValueError('# of frames must be a multiple of 2^dense_depth!')

    ### SPARSE COMPUTATION ###
    t0 = time.time()
    print('Computing sparse multitau G2t...')

    G2tmt = [[] for i in range(sparse_depth)]
    N_sparseloops = Itp.shape[0]//2**sparse_depth
    for N in tqdm(range(N_sparseloops)):
        Itp1 = Itp[N*2**sparse_depth:(N+1)*2**sparse_depth]
        Itp2 = Itp[(N+1)*2**sparse_depth:(N+2)*2**sparse_depth]

        # Compute central G2t
        G2t = _get_symG2t(Itp1)
        G2tmt = _G2t2G2tmt(G2t, G2tmt, type='sym')
        
        # Compute shifted G2t
        if N != N_sparseloops-1:
            G2t = _get_nonsymG2t(Itp1, Itp2)
            G2tmt = _G2t2G2tmt(G2t, G2tmt, type='non-sym')

    # Concatenate the G2tmt
    for i in range(len(G2tmt)): G2tmt[i] = np.concat(G2tmt[i])
    print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    ### DENSE COMPUTATION ###
    if dense_depth>sparse_depth:
        t0 = time.time()
        print('Computing dense multitau G2t...')
        # BIN Itp by 2^sparse_depth
        BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0]), (np.arange(Itp.shape[0])//2**sparse_depth, np.arange(Itp.shape[0]))), dtype=np.float32)
        Itp = dot_product_mkl(BIN_matrix, Itp, dense=True)

        # recursevly compute G2t first diagonal and bin by a factor 2
        for i in tqdm(range(dense_depth-sparse_depth)):
            G2t = (Itp[:-1] * Itp[1:]).sum(axis=1)  # G2t = <Itp*Itp(t-shifted)>p
            norm = np.sqrt(Itp.shape[1])/Itp.sum(axis=1) # standard normalization
            G2t = G2t * norm[1:] * norm[:-1]
            G2tmt.append(np.array(G2t))

            # Vin Itp by a factor 2
            BIN_matrix = sparse.csr_array((np.ones(Itp.shape[0]), (np.arange(Itp.shape[0])//2, np.arange(Itp.shape[0]))), dtype=np.float32)
            Itp = dot_product_mkl(BIN_matrix, Itp, cast=True)

        print('Done! (elapsed time =', round(time.time()-t0, 2), 's)')

    return G2tmt

#############################
##### PLOT multitau G2t #####
#############################

def plot_G2tmt(G2tmt, itime, vmin, vmax, lower_mt=4, yscale='log2', filter_layer=None, borders=False, xlims=None, vlines=None):

    if borders: linewidth = .2
    else:       linewidth = 0

    plt.figure(figsize=(10,5))
    T = (G2tmt[0].shape[0]+1) * itime
    for b in range(lower_mt, len(G2tmt)):
        if G2tmt[b].size>1:
            xmin = np.arange(2**b*2, (G2tmt[0].shape[0]+1), 2**b)*itime - itime*2**b/2
            xmin = np.insert(xmin, 0, 0)
            xrange = np.ones(G2tmt[b].size-2) * itime*2**b
            xrange = np.insert(xrange, 0, itime*2**b*3/2)
            xrange = np.insert(xrange, xrange.size, itime*2**b*3/2)
        else:
            xmin = np.array([0])
            xrange = np.array([T])

        xranges = [(xmin[i],xrange[i]) for i in range(len(xmin))]

        if (yscale=='log') or (yscale=='lin'): yrange = (2**b*itime, 2**b*itime)
        elif yscale=='log2':                   yrange = (b, 1)

        if vlines != None:
            for vline in vlines:
                plt.axvline(x=vline, color='red', linestyle='--', linewidth=1)
        
        if (filter_layer == None) or (b>=filter_layer): BB = plt.broken_barh(xranges, yrange, array=G2tmt[b],                                                         cmap='viridis', clim=(vmin, vmax), edgecolor='black', linewidth=linewidth)
        else:                                           BB = plt.broken_barh(xranges, yrange, array=gaussian_filter1d(G2tmt[b], 2**(filter_layer-b), mode='nearest'), cmap='viridis', clim=(vmin, vmax), edgecolor='black', linewidth=linewidth)

    plt.xlabel('$t_0$ [s]')                                                                 # x-axis label
    if xlims==None: plt.xlim(0, T)                                                          # x-axis limits (default)
    else:           plt.xlim(xlims)                                                         # or user defined
    if (yscale=='log') or (yscale=='lin'): plt.ylabel('$\\Delta T$ [s]')                    # y-axis label for linear and log scale
    elif yscale=='log2':                   plt.ylabel('$\\Delta T$ [$\\log_2$]')            # or log2 scale
    if (yscale=='log') or (yscale=='lin'): plt.ylim(2**lower_mt*itime, 2**len(G2tmt)*itime) # y-axis limits for linear and log scale
    elif yscale=='log2':                   plt.ylim(lower_mt, len(G2tmt))                   # or log2 scale
    if yscale == 'log':                    plt.yscale('log')                                # set y-axis to log scale (if yscale is log)
    plt.colorbar(BB)                                                                        # colorbar      
    plt.tight_layout(); plt.show()


#############################
##### GET TIMES 4 G2tmt #####
#############################

def get_t_G2tmt(itime, G2tmt):
    return [np.arange(itime*2**b, (G2tmt[0].shape[0]+1) * itime, itime*2**b) for b in range(len(G2tmt))]

def get_dt_G2tmt(itime, G2tmt):
    return np.array([itime*2**b for b in range(len(G2tmt))])


###############################
##### GET g2mt from G2tmt #####
###############################

def get_g2mt(itime, G2tmt):
    t_g2mt = 2**np.arange(len(G2tmt))*itime
    g2mt = np.array([np.mean(G2tmt[b]) for b in range(len(G2tmt))])
    dg2mt = np.array([np.std(G2tmt[b])/np.sqrt(G2tmt[b].size) for b in range(len(G2tmt))])
    return t_g2mt, g2mt, dg2mt


###################################
##### CUT G2tmt @ (tmin,tmax) #####
###################################

def cut_G2tmt(itime, G2tmt, tmin=None, tmax=None):
    G2tmt_cut = []
    for b in range(len(G2tmt)):
        if tmin == None: tmin = 0
        if tmax == None: tmax = (G2tmt[0].shape[0]+1) * itime

        sel = (np.arange(2**b, G2tmt[0].shape[0]+1, 2**b)*itime - itime*2**b >= tmin) * (np.arange(2**b, G2tmt[0].shape[0]+1, 2**b)*itime + itime*2**b <= tmax)
        if sel.sum() == 0:
            return G2tmt_cut
        else:
            G2tmt_cut.append(G2tmt[b][sel])
    return G2tmt_cut