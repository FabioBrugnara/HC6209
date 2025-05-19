"""
ID10_tools
==========

A python library for loading and processing data from the ID10 beamline at the European Synchrotron Radiation Facility (ESRF). The library provides functions for loading and processing data from the Eiger 4M detector, including converting dense data to sparse format, loading scan information, and handling overflow values.

Author: Fabio Brugnara
"""

### IMPORT SCIENTIFIC LIBRARIES ###
import os
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt

### IMPORT OTHER LIBRARIES ###
import h5py # for hdf5 file reading
import hdf5plugin # for hdf5 file reading
from joblib import Parallel, delayed    # for parallel processing

##### FILE PARAMETERS ######
Nx = 2162
Ny = 2068
Npx = Nx*Ny
############################

##### DETECTOR DIMENSION #####
lxp, lyp = 75e-6, 75e-6 #m
##############################


######## OF MASK OF LINES OF E4M DETECTOR ##########
OF = np.zeros((Nx, Ny), dtype=bool)
OF[512:550, :]   = True
OF[1062:1100, :] = True
OF[1612:1650, :] = True
OF[:, 1028:1040] = True
OF[:, 513:515]   = True
OF[:, 1553:1555] = True
OF = OF.flatten()
####################################################

###################
### SET VERSION ###
###################

def set_version(v):
    '''
    This function set some parameters for using a version of the ID10 line. We can add here as many versions as we want. The version v1 is the one used in the ID10 line before 2023. The v2 version is the one used in the ID10 line after 2023. The function set the parameters for the version selected.

    Parameters
    ----------
        v: string
            Version of the ID10 line ('v1', 'v2', or new others ...)
    '''
    global V, Nfmax_dense_file, Nfmax_sparse_file, len_dataset_string, len_scan_string, len_fileidx_string, df_eiger4m_entry
    if v=='v1':
        V = 'v1'
        Nfmax_dense_file = 2000 # this is the default value
        len_dataset_string = len_scan_string = 4
        len_fileidx_string = 4
        df_eiger4m_entry = 'eiger4m'

    elif v=='v2':
        V = 'v2'
        Nfmax_dense_file = 2000 # this is the default value
        Nfmax_sparse_file = 10000 # this is the default value
        len_dataset_string = 4
        len_scan_string = 4
        len_fileidx_string = 5
        df_eiger4m_entry = 'eiger4m_v2'

    ### add other versions here !!!

    else:
        raise ValueError('Version not recognized!')


####################################
####### LOAD SCAN INFO #############
###################################b#

def load_scan(raw_folder, sample_name, Ndataset, Nscan):
    '''
    Load scan parameters from h5 file. Many try-except are used to avoid errors in the case of missing parameters. The function will try to load the parameters and if they are not present, it will skip them.
    Fill free to add other parameters to the scan dictionary, with the try-except method.
    
    Parameters
    ----------
    raw_folder: string
        path to raw data folder
    sample_name: string
        name of the sample
    Ndataset: int
        number of the dataset
    Nscan: int
        number of the scan

    Returns
    -------
    scan: dict
        dictionary with scan parameters
    '''

    # LOAD H5 FILE
    h5file = h5py.File(raw_folder + sample_name+'/' +sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'/' + sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'.h5', 'r')[str(Nscan)+'.1']

    # LOAD SCAN PARAMETERS
    scan = {}
    scan['command'] = h5file['title'][()].decode("utf-8")
    scan['start_time'] = h5file['start_time'][()].decode("utf-8")

    try: scan['end_time'] = h5file['end_time'][()].decode("utf-8")
    except: pass

    try: scan['fast_timer_trig'] = h5file['measurement']['fast_timer_trig'][:]
    except: pass
    try: scan['fast_timer_period'] = h5file['measurement']['fast_timer_period'][:]
    except: pass

    try: scan['elapsed_time'] = h5file['measurement']['elapsed_time'][:]
    except: pass

    try: scan['xray_energy'] = h5file['instrument']['metadata']['eiger4m']['xray_energy'][()]
    except: pass
    try: scan['monoe'] = h5file['instrument']['positioners']['monoe'][()]
    except: pass

    try: scan['eh2diode'] = h5file['measurement']['eh2diode'][:]
    except: pass
    try: scan['ch2_saxs'] = h5file['measurement']['ch2_saxs'][:]
    except: pass
    try: scan['ys'] = h5file['measurement']['ys'][:]
    except: pass
    try: scan['zs'] = h5file['measurement']['zs'][:]
    except: pass
    try: scan['delcoup'] = h5file['instrument']['positioners']['delcoup'][:]
    except: scan['delcoup'] = h5file['instrument']['positioners']['delcoup'][()]
    try: scan['current'] = h5file['measurement']['current'][:]
    except: pass
    try: scan['mon'] = h5file['measurement']['mon'][:]
    except: pass
    try: scan['det'] = h5file['measurement']['det'][:]
    except: pass

    return scan


#############################
###### LOAD PILATUS #########
#############################

def load_pilatus(raw_folder, sample_name, Ndataset, Nscan, Nfi=None, Nff=None, Lbin=None):
    '''
    Load pilatus images from h5 file.
    
    Work in progress
    1) work with multiple files?
    2) directly load files from the directory (not relayng on the master hdf5 file)

    Parameters
    ----------
    raw_folder: string
        path to raw data folder
    sample_name: string
        name of the sample
    Ndataset: int
        number of the dataset
    Nscan: int
        number of the scan
    Nfi: int
        number of the first image
    Nff: int
        number of the last image

    Returns
    -------
    pilatus: dict
        dictionary with pilatus images
    '''

    if Lbin is None: Lbin = 1

    # LOAD H5 FILE
    h5file = h5py.File(raw_folder + sample_name+'/' +sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'/' + sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'.h5', 'r')[str(Nscan)+'.1']

    # LOAD PILATUS IMAGES
    pilatus_data = h5file['measurement']['pilatus300k'][Nfi:Nff]

    # BIN IMMAGES (IF Lbin>1)
    if Lbin > 1:
        pilatus_data = pilatus_data[0:pilatus_data.shape[0]//Lbin*Lbin]
        pilatus_data = pilatus_data.reshape((pilatus_data.shape[0]//Lbin, Lbin, pilatus_data.shape[1], pilatus_data.shape[2])).sum(axis=1)

    return pilatus_data


#######################################
######### LOAD E4M DATA ###############
#######################################

def load_dense_e4m(raw_folder, sample_name, Ndataset, Nscan, Nfi, Nff,  n_jobs=6, mask=None, tosparse=False):
    '''
    Load all the e4m data present in a scan.
    If tosparse=True (default) convert the dataframes to a sparse array.
    In older versions of the line ('v1') many overflow values are present in these frames, as they rapresent the lines on the detector, and also burned pixels. 
    To save mamory, we want to avoid saving these values within the sparse array, as they are likely to be the same in all frames. 
    The function generate an image (frame) of the overflow values selecting the pixel that are in overfllows in all the first Nf4overflow frames. 
    This image, called OF, can be then used to mask the overflow values in the sparse array.
    
    Work in progress
        1) Now this function doesn't work for version 'v1'!

    Future perspectives
        1) Nstep myght be usefull.

    Parameters
    ----------
        raw_folder: string
            the folder where the raw data is stored
        file_name: string
            the name of the file
        Nscan: int
            the scan number
        n_jobs: int
            number of parallel jobs to use (default=6)
        tosparse: bool
            if True return a sparse array, otherwise return a numpy array (default=True)
        Nf4overflow: int
            the number of frames to use to generate the overflow image (default=10)
    
    Returns
    -------
        OF: np.array
            the overflow image (ONLY FOR OLDER ID10 VERSIONS ('v1')!!!)
        sA: scipy.sparse.csr_array
            the sparse array with the e4m data (shape: Nf x Npx)
    
    '''

    print('Loading dense array ...')

    ### E4M DATA FOLDER
    h5_folder = raw_folder + sample_name+'/' +sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'/' + 'scan'+str(Nscan).zfill(len_dataset_string) + '/'

    #### GET THE INDEX (number) OF THE FILES THAT SHOULD BE LOADED
    if Nfi is None: file_i = 0 # if Nfi is None, start from the file 0
    else: file_i = Nfi//Nfmax_dense_file
    if Nff is None: file_f = len([i for i in os.listdir(h5_folder) if i.startswith('eiger4m_v2_frame_0_')]) # if Nff is None, finish at the last file
    else : file_f = Nff//Nfmax_dense_file+1


    ### LOAD FRAMES BY FILE FUNCTION (file i, with Nfi, Nff refered to the full run)
    def load_framesbyfile(i, Nfi, Nff, file_i, file_f): # function to loop over files in multiprocessing
        # IMPORT HDF5PLUGIN (needed for the parallel loop)
        import hdf5plugin

        # LOAD H5 FILE
        if V=='v1':
            print(   '\t -> loading file', 'eiger4m_'            + str(i).zfill(len_fileidx_string) + '.h5', '('+str(i+1)+'/'+str(file_f-file_i)+' loops)')
            h5file = h5py.File(h5_folder + 'eiger4m_'            + str(i).zfill(len_fileidx_string) + '.h5', 'r')
        elif V=='v2':
            print(   '\t -> loading file', 'eiger4m_v2_frame_0_' + str(i).zfill(len_fileidx_string) + '.h5', '('+str(i+1)+'/'+str(file_f-file_i)+' loops)')
            h5file = h5py.File(h5_folder + 'eiger4m_v2_frame_0_' + str(i).zfill(len_fileidx_string) + '.h5', 'r')

        # GET DATA
        if (file_i==i)  and (Nfi is not None):   a = Nfi%Nfmax_dense_file
        else: a = 0
        if (file_f==i+1) and (Nff is not None):  b = Nff%Nfmax_dense_file
        else: b = Nfmax_dense_file

        # LOAD DATA
        npA = h5file['entry_0000']['measurement']['data'][a:b].reshape((b-a, Npx))

        if V=='v1':
            npA[:,OF] = 0 # remove overflow

        if mask is not None: npA = npA[:,mask] # apply mask

        if tosparse: return sparse.csr_array(npA) # fill sparse array list
        else:        return npA # or fill numpy array list

    #  PARALLEL LOOP (gose from file_i to file_f)
    A = Parallel(n_jobs=n_jobs, require='sharedmem')(delayed(load_framesbyfile)(i, Nfi, Nff, file_i, file_f) for i in range(file_i, file_f))
    print('Done!')

    #### CONACATENATE THE RESULT ####
    print('Concatenating vectors ...')
    if not tosparse: A = np.vstack(A)
    else: A = sparse.vstack(A)
    print('Done!')

    print('--------------------------------------------------------')
    if tosparse:       print('Sparsity:    ', '{:.1e}'.format(A.size/(A.shape[0]*A.shape[1])))
    if tosparse:       print('Memory usage (scipy.csr_array):', '{:.3f}'.format((A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)/1024**3), 'GB', '(np.array usage:', '{:.3f}'.format(A.shape[0]*A.shape[1]*4/1024**3), 'GB)')
    elif not tosparse: print('Memory usage (numpy.array):', '{:.3f}'.format(A.nbytes/1024**3), 'GB')
    print('--------------------------------------------------------')

    return A


#######################################
######## LOAD E4M SPARSE ARRAY ########
#######################################

def load_sparse_e4m(raw_folder, sample_name, Ndataset, Nscan, Nfi=None, Nff=None, mask=None, n_jobs=10):
    '''
    Load the sparse array and the overflow image from the correct e4m raw_data folder.
    This function works differently depending on the version of the ID10 line used. 
    In the older version ('v1') the data should be first converted into the sparse format with the function ID10_tools.convert_dense_e4m.
    In the new version ('v2') the data is already saved in a sparse format at the line.
    
    Future perspectives
    1) implement Nstep to load only a part of the data
    
    Parameters
    ----------
        raw_folder: string
            the folder where the raw data is stored
        sample_name: string
            the name of the sample
        Ndataset: int: 
            the number of the dataset
        Nscan: int
            the scan number
        Nfi: int  
            the first frame to load (ONLY FOR THE V2 VERSION!) (default=None)
        Nff: int
            the last frame to load (ONLY FOR THE V2 VERSION!) (default=None)
        mask: np.array
            a mask that allow to select the pixels to load (ONLY FOR THE V2 VERSION!) (default=None)
        n_jobs: int
            number of parallel jobs to use (default=10)
    Returns
    -------
        OF: np.array
            the overflow image (ONLY FOR OLDER ID10 VERSIONS ('v1')!!!)
        sA: scipy.sparse.csr_array
            the sparse array with the e4m data (shape: Nf x Npx)
    '''
    ### V1 VERSION ###
    # The data are loaded from a single file which has been generated by this library (npz format) using the function ID10_tools.save_sparse_e4m_v1. The overflow image is saved in a separate file (npy format).
    # With this version the overflow image should be also loaded from the file.
    if V=='v1':
        e4m_sparse_file   = raw_folder + sample_name+'/' +sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'/scan'+str(Nscan).zfill(len_scan_string)+'/eiger4m_sparse.npz'
        e4m_overflow_file = raw_folder + raw_folder + sample_name+'/' +sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'/scan'+str(Nscan).zfill(len_scan_string)+'/eiger4m_overflow.npy'

        print('Loading sparse array ...')    
        sA = sparse.load_npz(e4m_sparse_file).tocsr()
        print('\t | Sparse array loaded from', e4m_sparse_file)
        print('\t | Sparsity:    ', '{:.1e}'.format(sA.size/(sA.shape[0]*sA.shape[1])))
        print('\t | Memory usage (scipy.csr_array):', '{:.3f}'.format((sA.data.nbytes + sA.indices.nbytes + sA.indptr.nbytes)/1024**3), 'GB', '(np.array usage:', '{:.3f}'.format(sA.shape[0]*sA.shape[1]*4/1024**3), 'GB)')
        print('Done!')
        print('Loading overflow array ...')
        OF = np.load(e4m_overflow_file)
        print('\t | Overflow array loaded from', e4m_overflow_file)
        print('Done!')
        return OF, sA
    
    elif V=='v2':
        print('Loading sparse array ...')  
        #### E4M DATA FOLDER
        h5_folder =  raw_folder + sample_name+'/' +sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'/scan'+str(Nscan).zfill(len_scan_string)+'/'

        #### GET THE INDEX (number) OF THE FILES THAT SHOULD BE LOADED
        if Nfi is None: file_i = 0 # if Nfi is None, start from the file 0
        else: file_i = Nfi//Nfmax_sparse_file
        if Nff is None: file_f = len([i for i in os.listdir(h5_folder) if i.startswith('eiger4m_v2_sparse_frame_0_')]) # if Nff is None, finish at the last file
        else : file_f = Nff//Nfmax_sparse_file+1

        ### LOAD FRAMES BY FILE FUNCTION (file i, with Nfi, Nff refered to the full run)
        def load_framesbyfile(i, Nfi, Nff, file_i, file_f, mask=None):
                # IMPORT HDF5PLUGIN (needed for the parallel loop)
                import hdf5plugin

                # LOAD H5 FILE
                print(   '\t -> loading file', 'eiger4m_v2_sparse_frame_0_' + str(i).zfill(len_fileidx_string) + '.h5', '('+str(i+1)+'/'+str(file_f-file_i)+' loops)')
                h5file = h5py.File(h5_folder + 'eiger4m_v2_sparse_frame_0_' + str(i).zfill(len_fileidx_string) + '.h5', 'r')

                # GET DATA IN CSR FORMAT
                # load frame_ptr (use Nfi and Nff to get the correct frames)
                a, b = None, None
                if (file_i==i)  and(Nfi is not None): a = Nfi%Nfmax_sparse_file
                if (file_f==i+1)and(Nff is not None): b = Nff%Nfmax_sparse_file+1
                
                frame_ptr = h5file['entry_0000']['measurement']['data']['frame_ptr'][a:b]
                index =     h5file['entry_0000']['measurement']['data']['index']    [frame_ptr[0]:frame_ptr[-1]]
                intensity = h5file['entry_0000']['measurement']['data']['intensity'][frame_ptr[0]:frame_ptr[-1]]

                # If necessary, reset frame_ptr starting from 0 (has to do with the csr array creation)
                if (file_i==i) and (Nfi is not None): frame_ptr = frame_ptr-frame_ptr[0]
                
                out = sparse.csr_array((intensity, index, frame_ptr), shape=(frame_ptr.shape[0]-1, Npx))
                if mask is not None: out = out[:,mask]
                return  out
        # PARALLEL LOOP (gose from file_i to file_f)
        sA = Parallel(n_jobs=n_jobs)(delayed(load_framesbyfile)(i, Nfi, Nff, file_i, file_f, mask) for i in range(file_i, file_f))
        print('Done!')

        #### CONACATENATE THE RESULT ####
        print('Concatenating vectors ...')
        sA = sparse.vstack(sA)
        print('Done!')


        print('\t | Sparse array loaded from', h5_folder)
        print('\t | Shape:      ', sA.shape)
        print('\t | Sparsity:    ', '{:.1e}'.format(sA.size/(sA.shape[0]*sA.shape[1])))
        print('\t | Memory usage (scipy.csr_array):', '{:.3f}'.format((sA.data.nbytes + sA.indices.nbytes + sA.indptr.nbytes)/1024**3), 'GB', '(np.array usage:', '{:.3f}'.format(sA.shape[0]*sA.shape[1]*4/1024**3), 'GB)')

        return sA

    else: print('Version not recognized! Please set the version with ID10_tools.set_version(v).\n')



###############################################################################################################
#######################################    ONLY V1 VERSION FUNCTIONS    #######################################
###############################################################################################################

#################################
######### GET NBIT (v1) #########
#################################

def get_Nbit_v1(raw_folder, sample_name, Ndataset, Nscan):
    '''
    Get the number of bits of the e4m data in the master file.
    The function loads the first image from the first file and check the maximum value.
    The maximum value is used to determine the number of bits.

    Parameters
    ----------
    raw_folder: string
        path to raw data folder
    sample_name: string
        name of the sample
    Ndataset: int
        number of the dataset
    Nscan: int
        number of the scan

    Returns
    -------
    Nbit: int
        number of bits of the e4m data
    '''

    # LOAD MASTER DATASET FILE
    e4m_file = raw_folder + sample_name+'/' +sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'/' + sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'.h5'
    h5file = h5py.File(e4m_file, 'r')[str(Nscan)+'.1']

    # LOAD EIGER4M DF LINK
    df = h5file['measurement'][df_eiger4m_entry]

    # TAKE THE FIRST IMAGE FROM THE FIRST FILE
    npA = df[0]

    # FIND THE NUMBER OF BITS
    if npA.max() == 2**16-1: Nbit = 16
    elif npA.max() == 2**32-1: Nbit = 32
    elif npA.max() == 2**8-1: Nbit = 8
    else: Nbit = 8

    return Nbit


############################################
######## SAVE E4M SPARSE ARRAY (V1) ########
############################################

def save_sparse_e4m_v1(OF, sA, raw_folder, sample_name, Ndataset, Nscan):
    '''
    Save the sparse array and the overflow image in the correct e4m raw_data folder. This function is usefull only for the older version of the ID10 line ('v1'). In the new version ('v2') the data is already saved in a sparse format.

    Future perspectives
    1) save sparse array in multiple files to load them faster in parallel\n

    Parameters
    ----------
        OF: np.array
            the overflow image
        sA: scipy.sparse.csr_array
            the sparse array with the e4m data (shape: Nf x Npx)
        raw_folder: str
            the folder where the raw data is stored
        sample_name: str
            the name of the sample
        Ndataset: int
            the number of the dataset
        Nscan: int
            the scan number
            
    '''

    e4m_sparse_file   = raw_folder + sample_name+'/' +sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'/scan'+str(Nscan).zfill(len_scan_string)+'/eiger4m_sparse.npz'
    e4m_overflow_file = raw_folder + sample_name+'/' +sample_name+'_'+str(Ndataset).zfill(len_dataset_string)+'/scan'+str(Nscan).zfill(len_scan_string)+'/eiger4m_overflow.npy'

    # Save the sparse array
    print('Saving sparse array ...')
    sparse.save_npz(e4m_sparse_file, sA, compressed=False)
    print('\t -> Sparse array saved in:', e4m_sparse_file)
    print('Done!')

    # Save the overflow image
    print('Saving overflow array ...')
    np.save(e4m_overflow_file, OF)
    print('\t -> Overflow array saved in:', e4m_overflow_file)
    print('Done!')


#################################################
######## CONVERT E4M DATA 2 SPARSE (V1) #########
#################################################

def convert_dense_e4m_v1(raw_folder, sample_name, Ndataset, Nscan, n_jobs=6, of_value=None, Nf4overflow=10,):
    '''
    Convert the e4m data in the master file to a sparse array. The function generate an image (frame) of the overflow values selecting the pixel that are in overfllows in all the first Nf4overflow frames. This image, called OF, can be then used to mask the overflow values in the sparse array.

    Parameters
    ----------
        raw_folder: str
            the folder where the raw data is stored
        file_name: str
            the name of the file
        Nscan: int
            the scan number
        Nf4overflow: int
            the number of frames to use to generate the overflow image (default=10)
    
    Returns
    -------
        OF: np.array
            the overflow image#################
        sA: scipy.sparse.csr_array
            the sparse array with the e4m data (shape: Nf x Npx)
    '''

    print('CONVERTING '+sample_name+', DATASET '+str(Ndataset).zfill(len_dataset_string)+', SCAN '+str(Nscan).zfill(len_scan_string)+' TO SPARSE ARRAY ...\n')
    OF, sA = load_dense_e4m(raw_folder, sample_name, Ndataset, Nscan, tosparse=True, Nf4overflow=Nf4overflow, n_jobs=n_jobs, of_value=of_value)
    save_sparse_e4m_v1(OF, sA, raw_folder, sample_name, Ndataset, Nscan)
    print('\nDONE!')
    return None







