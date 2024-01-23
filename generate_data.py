"""
@ File: generate_data.py
@ Author: Cunliang Ma and Sen Wang et al
@ Email: mcl@jxust.edu.cn
@ Brief: This is the main script that generate the data.
@ Copyright(C): 2023 Jiangxi University of Science and Technology. All rights reserved
"""

import argparse
import gc
import time
import os

import matplotlib

from tools.readconfig import readconfig
import h5py
import numpy as np
from additional_tools import gw_event_gps_list
from additional_tools import get_strain_file_list
from additional_tools import data_generation
from additional_tools import signal_padding_zero
from tools.pic_cqt import pic_cqt
import matplotlib.pyplot as plt
if __name__ == '__main__':
    matplotlib.use('TkAgg')
    parser = argparse.ArgumentParser(description='The name of the config file XXX.yml')
    parser.add_argument('--configfile', type=str, default='config.yml', help='--configfile default is config.yml')
    args = parser.parse_args()
    start_time = time.time()  # start of the time

    # the full path of the config file
    yaml_config_file = args.configfile
    yaml_config_path = os.path.join('.', 'config', yaml_config_file)

    # read the config file return a dictionary. The config file tell us:
    # the range of the mass, spin, snr, and the input_dir(strain without GW), output_dir
    config = readconfig(yaml_config_path)

    # analysis the config file, input dir (strain without GW) and output_dir
    input_dir = os.path.join('.', config['input_dir'])
    output_dir = os.path.join('.', config['output_dir'])

    # the strain file that without GW event in GWTC-1, GWTC-2.1, GWTC-3
    strain_file_list = get_strain_file_list(input_dir,gw_event_gps_list())

    # generate cut sub list
    sub_list = []  # every element contain the begin sub and end sub
    for sub in np.arange(4064):
        if sub % 4 == 0:
            sub_list.append(np.array([sub, (sub+32)], dtype=int))

    # the information that need be stored

    filenum = 0

    #loop to obtain the strain in hdf_file
    for strain_file in strain_file_list:
        strain_noise_list = []  # the strain of every sample
        psd_list = []  # the psd of every sample
        signal_list = []
        whitened_signal_list = []
        mass1_list = []
        mass2_list = []
        snr_list = []
        noise_plus_signal_list = []


        filenum=filenum+1
        num = 0
        print('The '+str(filenum)+'th file')
        h5_strain_file_path = os.path.join(input_dir, strain_file)
        h5_strain_information = h5py.File(h5_strain_file_path,'r')

        # strain information get
        GPS_time = np.array(h5_strain_information['meta']['GPSstart'])
        data_quality_mask = np.array(h5_strain_information['quality']['simple']['DQmask'], dtype=int)
        injection_mask = np.array(h5_strain_information['quality']['injections']['Injmask'], dtype=int)
        strain_data = np.array(h5_strain_information['strain']['Strain'])

        #data quality is good and no injection then cut 32 s
        data_quality_effect = np.ones(32)*127
        no_injection = np.ones(32)*23

        # loop to cut strain
        strain_cut = np.zeros(4096*32)

        for sub in sub_list:
            strain_cut = strain_data[sub[0]*4096:sub[1]*4096]
            if (np.all(data_quality_mask[sub[0]:sub[1]] == data_quality_effect) and  \
                    np.all(injection_mask[sub[0]:sub[1]] == no_injection) and  \
                    not np.any(np.isnan(strain_cut))):
                rescaled_signal, whiten_noise_strain, whiten_signal, whiten_signal_plus_noise, \
                mass1, mass2, target_snr, psd, is_rescaled = data_generation(strain_cut, config)

                if is_rescaled:
                    strain_noise_list.append(whiten_noise_strain[8*4096:24*4096])
                    psd_list.append(psd)
                    signal_list.append(rescaled_signal[8*4096:24*4096])
                    whitened_signal_list.append(whiten_signal[8*4096:24*4096])
                    mass1_list.append(mass1)
                    mass2_list.append(mass2)
                    snr_list.append(target_snr)
                    noise_plus_signal_list.append(whiten_signal_plus_noise[8*4096:24*4096])
                    num = num+1
                    # data can be used
                    del rescaled_signal, whiten_noise_strain, whiten_signal, whiten_signal_plus_noise, strain_cut,
                    gc.collect()
                    if num % 10 == 0:
                        print('filenum is '+str(filenum)+', and '+str(num)+'samples are generated')
                    #print('data can be used')
                else:
                    del rescaled_signal, whiten_noise_strain, whiten_signal, whiten_signal_plus_noise
                    gc.collect()

            else:
                #data cannot be used
                print('data cannot be used')


        strain_noise = np.array(strain_noise_list)
        psd = np.array(psd_list)
        signal = np.array(signal_list)
        whitened_signal = np.array(whitened_signal_list)
        mass1 = np.array(mass1_list)
        mass2 = np.array(mass2_list)
        snr = np.array(snr_list)
        noise_plus_signal = np.array(noise_plus_signal_list)
        output_file_name = os.path.join(output_dir,'compose_'+str(GPS_time)+'_'+str(num)+'.npz')
        np.savez(output_file_name,
                 strain_noise = strain_noise,
                 psd = psd,
                 signal = signal,
                 whitened_signal=whitened_signal,
                 mass1=mass1,
                 mass2=mass2,
                 snr=snr,
                 noise_plus_signal=noise_plus_signal)
                 
        del GPS_time, data_quality_mask, injection_mask, strain_data, h5_strain_information
        
        del strain_noise, psd, signal, whitened_signal, mass1, mass2, snr, noise_plus_signal
        
        del strain_noise_list, psd_list, signal_list, whitened_signal_list, mass1_list, mass2_list, snr_list,\
        noise_plus_signal_list
        gc.collect()