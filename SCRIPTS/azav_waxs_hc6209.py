import numpy as np
import time
import pyFAI
import pylab as plt
import pathlib
import h5py
from silx.io import h5py_utils
import hdf5plugin
import datastorage
from tqdm import tqdm
import configparser
import os
import sys


#PONIFILE = "pilatus_calib_15.875keV.poni"
PONIFILE = "lab6_eh2_exp_0001_scan0027.poni"
MASK = np.load("pilatus300k_mask_20240201.npy")
QNORM = 1.5,2


def extract_data(
    ai,
    fname,
    scan_num,
    mask=None,
    detector="pilatus300k",
    npt=2048,
    extra_data=None,
    save=None,
    nimages=None,
):
    stem = f"{pathlib.Path(fname).stem}_scan{scan_num:04d}"
    if extra_data is None:
        extra_data = []
    if isinstance(extra_data, str):
        extra_data = extra_data.split()
    extra_data = list(extra_data) + ["slow_epoch_trig",]# "elapsed_time"]
    if isinstance(ai, str):
        ai = pyFAI.load(ai)
    ret = dict()
    with h5py_utils.File(fname, mode="r") as h5:
        scan = h5[f"{scan_num}.1/measurement/"]
        for key in extra_data:
            ret[key] = scan[key][()]
        images = scan[detector]
        img_av = np.zeros(images.shape[1:])
        if nimages is None:
            nimages = len(images)
        azav = np.zeros((nimages, npt))
        for i in tqdm(range(nimages)):
            img = images[i]
            q, azav[i] = ai.integrate1d(
                img, npt, mask=mask, polarization_factor=-1, unit="q_A^-1"
            )
            img_av += img
        img_av = img_av / nimages
        sample_T = h5[f"{scan_num}.2/measurement/omega_sample"][()]
        body_T = h5[f"{scan_num}.2/measurement/omega_body"][()]
        epoch_T = h5[f"{scan_num}.2/measurement/epoch"][()]
        n = min(sample_T.size,body_T.size,epoch_T.size)
        sample_T_interp = np.interp(ret["slow_epoch_trig"],epoch_T[:n],sample_T[:n])
        body_T_interp = np.interp(ret["slow_epoch_trig"],epoch_T[:n],body_T[:n])
        norm_idx = (q>QNORM[0]) & (q<QNORM[1])
        norm = azav[:,norm_idx].mean(axis=1)
        norm = norm/norm.mean()
        ret["img_av"] = img_av
        ret["mask"] = mask
        ret["azav"] = azav
        ret["azav_norm"] = azav/norm[:,np.newaxis]
        ret["sample_T"] = sample_T_interp
        ret["body_T"] = body_T_interp
        ret["q"] = q
        ret["fname"] = stem

    ret = datastorage.DataStorage(ret)
    # if save == "auto":
    #    save = f"azav/{sample}_{detector}_dataset{dataset}_scan{scan_num}.h5"
    # print(save)
    if save == "auto":
        save = f"pilatus_azav/{stem}.h5"
    if save is not None:
        print(f"Saving data in {save}")
        ret.save(save)
    return ret



def do_scan(sample="ngl6_ac_sampl1",dataset=1,scan=10,nimages=None,mask=MASK):
    fname = f"../RAW_DATA/{sample}/{sample}_{dataset:04d}/{sample}_{dataset:04d}.h5"
    data = extract_data(
        PONIFILE,
        fname,
        scan,
        nimages=nimages,
        save="auto",
        npt=1024,
        mask=mask,
    )
    return data


def do_plot(data,**kw):
    nimages = min(data.sample_T.size,data.azav.shape[0])
    plt.pcolormesh(data.q,data.sample_T[:nimages],data.azav_norm[:nimages],**kw)
    plt.xlabel("q (Ang-1)")
    plt.ylabel("sample T")
    plt.title(data.fname)
    plt.grid()


def analysis_from_xpcs_template(xpcs_template="tmp_xpcs_input.txt"):
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(xpcs_template)
    outpout_folder = config['data_location']['result_dir']
    scan =   int(config['sample_description']['scan'].replace("scan",""))
    sample = config['sample_description']['name']
    dataset = int(config['sample_description']['dataset'])

    # read detector frame rate ratio from bliss file
    bliss_fname = f"../RAW_DATA/{sample}/{sample}_{dataset:04d}/{sample}_{dataset:04d}"
    h = h5py_utils.File(bliss_fname + ".h5","r")
    period_pilatus = h[f"{scan}.1/measurement/slow_timer_period"][0]
    period_eiger = h[f"{scan}.1/measurement/fast_timer_period"][0]
    h.close()
    ratio_pilatus_eiger = int(period_pilatus/period_eiger)

    # read azav
    azav_fname = "pilatus_azav/" + pathlib.Path(bliss_fname).stem + f"_scan{scan:04d}.h5"
    azav_fname = pathlib.Path(azav_fname)
    if not azav_fname.is_file():
        do_scan(sample=sample,dataset=dataset,scan=scan)
    data = datastorage.read(azav_fname)
    first_frame = int(config['data_location']['first_file']) // ratio_pilatus_eiger
    last_frame = int(config['data_location']['last_file']) // ratio_pilatus_eiger
    azav = data.azav[first_frame:last_frame].mean(axis=0)
    temp = data.sample_T[first_frame:last_frame].mean(axis=0)

    azav_fname = f"{outpout_folder}/{sample}_{dataset:04d}_pilatus_waxs.dat"
    np.savetxt(azav_fname,np.vstack((data.q,azav)).T)

    temp_fname = f"{outpout_folder}/{sample}_{dataset:04d}_temperature.dat"
    with open(temp_fname,"w") as f:
        f.write(f"{temp:.3f}")

    print(azav_fname)
    print(temp_fname)

if __name__ == "__main__" and len(sys.argv) > 1:
    xpcs_template = sys.argv[1]
    analysis_from_xpcs_template(xpcs_template)


#do_scan(sample="ngl6_ac_sampl1",dataset=1,scan=10)
#analysis_from_xpcs_template("tmp_xpcs_input.txt")
