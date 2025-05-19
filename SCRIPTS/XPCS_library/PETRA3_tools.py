### IMPORT SCIENTIFIC LIBRARIES ###
import numpy as np
import h5py
import hdf5plugin
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

##### FILE PARAMETERS ######
of_value = 2**16-1
Nx = 2167
Ny = 2070
Npx = Nx*Ny
Nfmax_file = 2000
len_scan_string = 5
###########################


# DETECTOR DIMENSION
lxp, lyp = 75e-6, 75e-6 #m


######## MASK OF LINES OF E4M DETECTOR ##########
e4m_mask = np.ones((Nx, Ny), dtype=bool)

# mask rows
e4m_mask[:, 1020:1048] = False
e4m_mask[505:561, :] = False
e4m_mask[1055:1111, :] = False
e4m_mask[1605:1665, :] = False

# mask borders
e4m_mask[0:30, :] = False
e4m_mask[-30:, :] = False
e4m_mask[:, 0:30] = False
e4m_mask[:, -30:] = False

e4m_mask = e4m_mask.reshape(-1)

#################################################


####### GET ATTENUATION #############
def _get_att(scan):
    att_dict = {(0,0):0, (1,0):1, (0,1):2, (1,1):3, (2,0):4, (0,2):5, (2,1):6, (1,2):7, (2,2):8}
    return att_dict[(round(scan['abs1z']), round(scan['abs2z']))]


########### READ E4M TIME ###########
def _read_e4m_itime(raw_folder, file_name, Nscan):
    e4m_file = raw_folder + file_name+'_'+str(Nscan).zfill(len_scan_string) + '/e4m/' + file_name+'_'+str(Nscan).zfill(len_scan_string)+'_master.h5'
    h5file = h5py.File(e4m_file, "r")
    return float(np.array(h5file['entry']['instrument']['detector']['frame_time']))

########### READ E4M WL ###########
def _read_e4m_wl(raw_folder, file_name, Nscan):
    e4m_file = raw_folder + file_name+'_'+str(Nscan).zfill(len_scan_string) + '/e4m/' + file_name+'_'+str(Nscan).zfill(len_scan_string)+'_master.h5'
    h5file = h5py.File(e4m_file, "r")
    return float(np.array(h5file['entry']['instrument']['beam']['incident_wavelenght']))


##########################################
########### LOAD SCAN FIO FILE ###########
##########################################

def load_scan(raw_folder, file_name, Nscan):
    '''
    Load fio file and return a dictionary with the following keys:
    - command: the command used to generate the file
    - params: a dictionary with the parameters used by the scan
    - data: a pandas dataframe with the data
    The .fio scan can be related both to a simple scan and to an e4mscan.
    
    Args:
        raw_folder (str): the folder where the raw data is stored
        file_name (str): the name of the file
        Nscan (int): the scan number
    
    Returns:
        fio_data (dict): a dictionary with the fio file data    
    '''
    # READ .fio file lines
    unknown_fio = True

    # e4mscan
    try:
        with open(raw_folder + file_name+'_'+str(Nscan).zfill(len_scan_string) + '/' + file_name+'_'+str(Nscan).zfill(len_scan_string)+'.fio') as f:
            fio_lines = f.readlines()
            fio_type = 'e4mscan'
            unknown_fio = False
            fio_file = raw_folder + file_name+'_'+str(Nscan).zfill(len_scan_string) + '/' + file_name+'_'+str(Nscan).zfill(len_scan_string)+'.fio'
    except: pass

    # simple scan
    try: 
        with open(raw_folder + file_name + '/' + file_name+'_'+str(Nscan).zfill(len_scan_string)+'.fio') as f:
            fio_lines = f.readlines()
            fio_type = 'simplescan'
            unknown_fio = False
            fio_file = raw_folder + file_name + '/' + file_name+'_'+str(Nscan).zfill(len_scan_string)+'.fio'
    except: pass

    if unknown_fio:
        print('.fio file not found! (The scan may not exist, or is a scan type saved in an unknown way)')
        return None

    def find_line(line, fio_lines):
        for i, l in enumerate(fio_lines):
            if l == line:
                return i
        return None

    # dic to save the data  in
    scan = {}

    # COMMAND (in e4mscan is not present, but still there is a %c line, followed by !...)
    scan['command'] = fio_lines[find_line('%c\n', fio_lines)+1].strip()

    # PARAMS
    # define last params line (depends wether there are data in the .fio)
    try: last_pline = find_line('%d\n', fio_lines)-3
    except: last_pline = len(fio_lines)-1
    # read the params
    for l in fio_lines[find_line('%p\n', fio_lines)+1:last_pline]:
        key, value = l.split('=')
        value = value.strip()
        key = key.strip()
        # if the key do not start with '_', than the value is a number
        if key[0]!='_':
            value=float(value)

        scan[key] = value

    # DATA
    # read data if present
    try:
        first_dline = find_line('%d\n', fio_lines)+1

        cols = []
        for l in fio_lines[first_dline:]:
            if l[:5] == ' Col ':
                cols.append(l.split()[2])
            else:
                first_dline = fio_lines.index(l)
                break

        scan['data'] = pd.read_csv(fio_file, skiprows=first_dline, skipfooter=1, names=cols, sep=' ', engine='python')
        scan['data'].reset_index(drop=True, inplace=True)
    except: pass


    # COMPUTE ATTENUATION
    scan['att'] = _get_att(scan)

    # GET E4M INTEGRATION TIME AND WAVELENGTH
    try:
        e4m_file = raw_folder + file_name+'_'+str(Nscan).zfill(len_scan_string) + '/e4m/' + file_name+'_'+str(Nscan).zfill(len_scan_string)+'_master.h5'
        h5file = h5py.File(e4m_file, "r")
        scan['itime'] = float(np.array(h5file['entry']['instrument']['detector']['frame_time']))
        scan['wl'] = float(np.array(h5file['entry']['instrument']['beam']['incident_wavelength']))
        scan['Ei'] = 12.39842/scan['wl']
    except: pass

    print('.fio LOADED! (fio type:', fio_type,', from:', fio_file, ')')    
    return scan


##################################
######### LOAD E4M TDATA #########
##################################

# THIS FUNCTION SHOULD BE PARALLELIZED IN THE FUTURE !!!

def load_e4m(raw_folder, file_name, Nscan, tosparse=True, Nf4overflow=10):
    '''
    Load all the e4m data present in a master file.
    If tosparse=True (default) convert the dataframes to a sparse array. Many overflow values are present in the frames (16 bit => 2^16-1), as they rapresent the lines on the detector, and also burned pixels. To save mamory, we want to avoid saving these values within the sparse array, as they are likely to be the same in all frames. The function generate an image (frame) of the overflow values selecting the pixel that are in overfllows in all the first Nf4overflow frames. This image, called OF, can be then used to mask the overflow values in the sparse array.

    Args:
        raw_folder (str): the folder where the raw data is stored
        file_name (str): the name of the file
        Nscan (int): the scan number
        tosparse (bool): if True return a sparse array, otherwise return a numpy array (default=True)
        Nf4overflow (int): the number of frames to use to generate the overflow image (default=10)
    
    Returns:
        OF (np.array): the overflow image
        sA (scipy.sparse.coo_array): the sparse array with the e4m data (shape: Nf x Npx)
    '''

    # load master file
    e4m_file = raw_folder + file_name+'_'+str(Nscan).zfill(len_scan_string) + '/e4m/' + file_name+'_'+str(Nscan).zfill(len_scan_string)+'_master.h5'
    print('Loading master hdf5 file ... ', e4m_file)
    print()
    h5file = h5py.File(e4m_file, "r")

    # print integration time and beam wavelength
    e4m_itime = float(np.array(h5file['entry']['instrument']['detector']['frame_time']))
    e4m_wl = float(np.array(h5file['entry']['instrument']['beam']['incident_wavelength']))
    print('Frame integration time:', '{:.1e}'.format(e4m_itime), 's')
    print('Beam wavelength:', '{:.2f}'.format(e4m_wl), 'A (', '{:.2f}'.format(12.39842/e4m_wl), 'keV )')

    # get data file names (linked in the master file)
    data_files = list(h5file['entry']['data'].keys())

    # Nf4overflow cant't be larger than the number of frames in the first data file
    if (Nf4overflow > h5file['entry']['data'][data_files[0]].shape[0]) or (Nf4overflow > Nfmax_file):
        print('Nf4overflow is too large !!!')
        print('used maximum Nf4overflow:', h5file['entry']['data'][data_files[0]].shape[0])

    # prepare OF array and list to store the sparse dataframes
    OF = np.zeros(Npx, dtype='bool')
    dfs = []

    print('# of frames :', (len(data_files)-1)*Nfmax_file + h5file['entry']['data'][data_files[len(data_files)-1]].shape[0])
    print('# of linked files:', len(data_files))
    print()

    print('Loading linked files ...')
    # cicle on the files
    for i in range(len(data_files)):
        print('\t -> loading linked file', data_files[i], '('+str(i+1)+'/'+str(len(data_files))+')')

        # load data in numpy
        try: df = h5file['entry']['data'][data_files[i]]
        except: break
        Nf_df = df.shape[0]
        npA = np.array(df).reshape((Nf_df, Npx))

        # generate OF array from the firsts images
        if i==0: OF = (npA[0:Nf4overflow]==of_value).all(axis=0)

        # remove overflow if tosparse=True
        npA[:,OF] =  0

        # fill sparse array list
        if tosparse: dfs.append(sparse.coo_array(npA))
        # or fill numpy array list
        else: dfs.append(npA)

    # concatenate sparse or nupy arrays
    if tosparse:
        sA = sparse.vstack(dfs)
    else:
        sA = np.concatenate(dfs)

    print('\n--------------------------------------------------------')
    if tosparse: print('Sparsity:    ', '{:.1e}'.format(sA.size/(sA.shape[0]*sA.shape[1])))
    else:        print('Sparsity:    ', '{:.1e}'.format(sA[sA!=0].size/sA.size))
    if tosparse: print('Memory usage:', '{:.1e}'.format((sA.data.nbytes + sA.row.nbytes + sA.col.nbytes)/1024/1024), 'MB', '(np.array usage:', '{:.1e}'.format(sA.shape[0]*sA.shape[1]*2/1024/1024), 'MB)')
    else:        print('Memory usage:', '{:.1e}'.format(sA.nbytes/1024/1024/1024), 'GB')
    print('--------------------------------------------------------')

    return OF, sA

#######################################
######## SAVE E4M SPARSE ARRAY ########
#######################################

def save_sparse_e4m(OF, sA, raw_folder, file_name, Nscan):
    '''
    Save the sparse array and the overflow image in the correct e4m raw_data folder.

    Args:
        OF (np.array): the overflow image
        sA (scipy.sparse.coo_array): the sparse array with the e4m data (shape: Nf x Npx)
        raw_folder (str): the folder where the raw data is stored
        file_name (str): the name of the file
        Nscan (int): the scan number
    '''

    e4m_sparse_file   = raw_folder + file_name+'_'+str(Nscan).zfill(len_scan_string) + '/e4m/' + file_name+'_'+str(Nscan).zfill(len_scan_string)+'_sparse.npz'
    e4m_overflow_file = raw_folder + file_name+'_'+str(Nscan).zfill(len_scan_string) + '/e4m/' + file_name+'_'+str(Nscan).zfill(len_scan_string)+'_overflow.npy'

    print('Saving sparse array ...')
    sparse.save_npz(e4m_sparse_file, sA, compressed=False)
    print('\t -> Sparse array saved in:', e4m_sparse_file)
    print('Done!')
    print('Saving overflow array ...')
    np.save(e4m_overflow_file, OF)
    print('\t -> Overflow array saved in:', e4m_overflow_file)
    print('Done!')

#######################################
######## LOAD E4M SPARSE ARRAY ########
#######################################

def load_sparse_e4m(raw_folder, file_name, Nscan):
    '''
    Load the sparse array and the overflow image from the correct e4m raw_data folder.

    Args:
        raw_folder (str): the folder where the raw data is stored
        file_name (str): the name of the file
        Nscan (int): the scan number
    
    Returns:
        OF (np.array): the overflow image
        sA (scipy.sparse.coo_array): the sparse array with the e4m data (shape: Nf x Npx)
    '''

    e4m_sparse_file   = raw_folder + file_name+'_'+str(Nscan).zfill(len_scan_string) + '/e4m/' + file_name+'_'+str(Nscan).zfill(len_scan_string)+'_sparse.npz'
    e4m_overflow_file = raw_folder + file_name+'_'+str(Nscan).zfill(len_scan_string) + '/e4m/' + file_name+'_'+str(Nscan).zfill(len_scan_string)+'_overflow.npy'
    
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

#######################################
######## CONVERT E4M 2 SPARSE #########
#######################################

def convert_e4m2sparse(raw_folder, sample_name, Ndataset, Nscan, Nf4overflow=10, n_jobs=6):
    '''
    Convert the e4m data in the master file to a sparse array. The function generate an image (frame) of the overflow values selecting the pixel that are in overfllows in all the first Nf4overflow frames. This image, called OF, can be then used to mask the overflow values in the sparse array.

    Args:
        raw_folder (str): the folder where the raw data is stored
        file_name (str): the name of the file
        Nscan (int): the scan number
        Nf4overflow (int): the number of frames to use to generate the overflow image (default=10)
    
    Returns:
        OF (np.array): the overflow image#################
        sA (scipy.sparse.csr_array): the sparse array with the e4m data (shape: Nf x Npx)
    '''

    print('CONVERTING '+sample_name+', DATASET '+str(Ndataset).zfill(len_dataset_string)+', SCAN '+str(Nscan).zfill(len_scan_string)+' TO SPARSE ARRAY ...\n')
    OF, sA = load_e4m(raw_folder, sample_name, Ndataset, Nscan, tosparse=True, Nf4overflow=Nf4overflow, n_jobs=6)
    save_sparse_e4m(OF, sA, raw_folder, sample_name, Ndataset, Nscan)
    print('\nDONE!')
    return None


