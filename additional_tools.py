"""
@ File: generate_data.py
@ Author: Cunliang Ma, Sen Wang, Wei Wang et al
@ Email: mcl@jxust.edu.cn
@ Brief: This is the main script that generate the data.
@ Copyright(C): 2023 Jiangxi University of Science and Technology. All rights reserved
"""
import gc
import random
import os

import numpy as np
import pycbc.catalog
import time
from pycbc.filter import matched_filter
from pycbc.types import TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import highpass
import sys
import gc
from memory_profiler import profile
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from tools.get_waveform import get_waveform

from tools.pic_cqt import pic_cqt

# run one times
"""
output the gwtc-1, gwtc-2.1 and gwtc-3 compact Binary coalescence event
"""
def gw_event_gps_list():
    event_gps_list = []
    c = [pycbc.catalog.Catalog(source='gwtc-1'),
         pycbc.catalog.Catalog(source='gwtc-2.1'),
         pycbc.catalog.Catalog(source='gwtc-3')]
    event_gps_list.extend([c[0][m].time for m in c[0]])
    event_gps_list.extend([c[1][m].time for m in c[1]])
    event_gps_list.extend([c[2][m].time for m in c[2]])

    return event_gps_list

"""
@input dir_name: dir name of the strain
       event_gps_list: GW event time list
@output strain file name list, the GW events is not included
"""
def get_strain_file_list(dir_name, event_gps_list):

    strain_file_list_not_clean = os.listdir(dir_name)
    strain_file_list_clean = []


    for strain_file_name in strain_file_list_not_clean:
        # pick the gps time
        file_gps_begin = float(strain_file_name.split('-')[-2])
        # is the strain contain a GW event
        iscontain = False
        for gps_time in event_gps_list:
            if gps_time >= file_gps_begin and gps_time <= file_gps_begin+4095:
                print(strain_file_name+' contain '+' GW event ')
                iscontain = True
                break
            else:
                continue
        if not iscontain:
            print(strain_file_name + ' not contain ' 'GW event')
            strain_file_list_clean.append(strain_file_name)

    print('before clean have '+ str(len(strain_file_list_not_clean)) + ' files')
    print('after clean have '+ str(len(strain_file_list_clean))+ ' files')
    return strain_file_list_clean

#input:hp
#output:extending the length of hp to length and max is at center
def signal_padding_zero(hp,noise_len):

    pos_max = np.argmax(hp)
    len_hp = hp.shape[0]

    if pos_max > noise_len/2:
        zero_append_list_after = np.zeros(int(noise_len/2)-(len_hp-pos_max))
        hp_new = hp[int(pos_max-noise_len/2):]
        appended_signal = np.concatenate([hp_new,zero_append_list_after], axis=0)
        del zero_append_list_after, hp_new,
        gc.collect()
        return appended_signal
    else:
        zero_append_list_before = np.zeros(int(noise_len / 2) - pos_max)
        zero_append_list_after = np.zeros(int(noise_len / 2) - (len_hp - pos_max))
        appended_signal = np.concatenate([zero_append_list_before, hp], axis=0)
        appended_signal = np.concatenate([appended_signal, zero_append_list_after], axis=0)
        del zero_append_list_after, zero_append_list_before
        gc.collect()
        return appended_signal

"""
input: the strain with numpy type
output: the psd of the strain
"""
def calculate_psd(strain):
    ts_strain = TimeSeries(strain, delta_t=1/4096.0)
    psd = ts_strain.psd(4)
    psd = interpolate(psd, ts_strain.delta_f)
    psd = inverse_spectrum_truncation(psd, 4*4096, low_frequency_cutoff=20)
    del ts_strain
    gc.collect()
    return psd


def whiten_strain(strain, return_psd=True):
    ts = TimeSeries(initial_array=strain, delta_t=1.0/4096)
    ts, psd = ts.whiten(segment_duration=4.0, max_filter_duration=4.0, remove_corrupted=False,
                   low_frequency_cutoff=20, return_psd=True)
    if(return_psd):
        return ts, psd
    else:
        return ts


def psd_whiten(signal, segment_duration, max_filter_duration, trunc_method='hann',
               remove_corrupted=True, low_frequency_cutoff=None,
               return_psd=False, psd=None, **kwds):
    """ Return a whitened time series

    Parameters
    ----------
    segment_duration: float
        Duration in seconds to use for each sample of the spectrum.
    max_filter_duration : int
        Maximum length of the time-domain filter in seconds.
    trunc_method : {None, 'hann'}
        Function used for truncating the time-domain filter.
        None produces a hard truncation at `max_filter_len`.
    remove_corrupted : {True, boolean}
        If True, the region of the time series corrupted by the whitening
        is excised before returning. If false, the corrupted regions
        are not excised and the full time series is returned.
    low_frequency_cutoff : {None, float}
        Low frequency cutoff to pass to the inverse spectrum truncation.
        This should be matched to a known low frequency cutoff of the
        data if there is one.
    return_psd : {False, Boolean}
        Return the estimated and conditioned PSD that was used to whiten
        the data.
    kwds : keywords
        Additional keyword arguments are passed on to the `pycbc.psd.welch` method.

    Returns
    -------
    whitened_data : TimeSeries
        The whitened time series
    """

    # Estimate the noise spectrum
    # psd = self.psd(segment_duration, **kwds)
    # psd = interpolate(psd, self.delta_f)
    signal1 = TimeSeries(signal, delta_t=1/4096.0)
    max_filter_len = int(max_filter_duration * signal1.sample_rate)
    #
    # # Interpolate and smooth to the desired corruption length
    # psd = inverse_spectrum_truncation(psd,
    #                                   max_filter_len=max_filter_len,
    #                                   low_frequency_cutoff=low_frequency_cutoff,
    #                                   trunc_method=trunc_method)

    # Whiten the data by the asd
    white = (signal1.to_frequencyseries() / psd ** 0.5).to_timeseries()

    if remove_corrupted:
        white = white[int(max_filter_len / 2):int(len(signal) - max_filter_len / 2)]


    del signal1, max_filter_len
    gc.collect()
    if return_psd:
        return white, psd

    return white

"""
input: signal, noise_strain, noise_psd, target_snr
output: rescaled_signal, whitened_strain, whitened_signal,  signal_plus_strain, scale
attention: the type of the signal is pycbc.types.TimeSeries
            the type of the noise_strain is np.array
"""
def rescale_signal_and_whiten(signal, noise_strain, noise_psd, target_snr):


    template_origin = TimeSeries(signal,delta_t=1/4096.0)

    # print('1st plot template')
    # plt.plot(np.array(template_origin))
    # plt.show()

    # os.system('pause')
    padding_signal = signal_padding_zero(np.array(template_origin), 32 * 4096)


    template_origin.resize(len(noise_strain))
    """
        debug to check the data
        """
    # print('second plot template resized')
    # plt.plot(np.array(template_origin))
    # plt.show()
    #
    # os.system('pause')

    template= template_origin.cyclic_time_shift(template_origin.start_time)

    # print('second plot template cyclisted')
    # plt.plot(np.array(template))
    # plt.show()
    #
    # os.system('pause')

    try_injected =TimeSeries( noise_strain + padding_signal,delta_t=1/4096.0)


    snr = matched_filter(template=template, data=try_injected, psd=noise_psd, low_frequency_cutoff=20)


    # print(snr)
    # print('plot try injected')
    # plt.plot(np.array(try_injected))
    # plt.show()
    # os.system('pause')

    scale_max_snr = np.max(np.array(abs(snr[4096*15:17*4096])))
    scale = target_snr
    rescaled_signal = padding_signal/scale_max_snr
    rescaled_injection = rescaled_signal+noise_strain
    max_snr = scale_max_snr

    is_rescaled = False

    for j in range(100):
        # if max_snr > target_snr:
        #     break
        #     print('scale is  '+ str(scale))
        # else:
        #     print('scale is '+str(scale))
        #     rescaled_signal = rescaled_signal * scale

        rescaled_signal_mid = rescaled_signal * scale
        del rescaled_signal
        rescaled_signal = rescaled_signal_mid
        # rescaled_signal = rescaled_signal*scale
        del rescaled_signal_mid
        rescaled_injection = TimeSeries(rescaled_signal+noise_strain, delta_t=1/4096.0)

        rescaled_snr = matched_filter(template=template, data=rescaled_injection,psd=noise_psd,low_frequency_cutoff=20)
        max_snr = np.max(np.array(abs(rescaled_snr[15*4096:17*4096])))
        del rescaled_snr
        scale = target_snr/max_snr
        #print('target snr is ' + str(target_snr) + ' rescaled snr is ' + str(max_snr))
        if abs(max_snr-target_snr) <= 0.1:
            #print('target snr and rescaled snr is smaller than 0.1')
            is_rescaled = True
            break




    #print('success rescaled')
    scale = scale/scale_max_snr
    whiten_signal = psd_whiten(rescaled_signal,
                                    segment_duration=4.0,
                                    max_filter_duration=4,
                                    remove_corrupted=False,
                                    psd=noise_psd,
                                    low_frequency_cutoff=20)

    whiten_signal_plus_strain = whiten_strain(rescaled_injection,
                                    return_psd=False)

    del template_origin, template, padding_signal, try_injected, snr,
    gc.collect()

    return rescaled_signal, whiten_signal, whiten_signal_plus_strain, scale, is_rescaled






"""
@ input
  strain: the 32 s strain
  config: the config dictionary 
@ output
  whitened_noise_strain   whitened_signal  signal
  mass1  mass2 snr 
"""
# @profile()
def data_generation(strain, config):

    #print('in data_generation')

    mass1 = config['max_mass1']
    mass2 = config['min_mass2']
    random.seed(time.time())
    mass_effect = False


    """
    random generate the parameter
    """
    # mass1 and mass2 must have the relation mass1>mass2 and mass1<10*mass2
    while not mass_effect:
        mass1 = random.uniform(config['min_mass1'],config['max_mass1'])
        mass2 = random.uniform(config['min_mass2'],mass1)
        if mass1 < 10*mass2:
            mass_effect = True
            break
    spin1z = random.uniform(config['min_spin1z'], config['max_spin1z'])
    spin2z = random.uniform(config['min_spin2z'], config['max_spin2z'])
    declination = random.uniform(0, np.pi)
    right_ascension = random.uniform(0, np.pi*2)
    polarization = random.uniform(0,np.pi*2)
    coa_phase = random.uniform(0,np.pi*2)
    target_snr = random.uniform(config['snr_min'],config['snr_max'])

    """
    generate the waveform
    """
    signal_h1, signal_l1, signal_v1, signal_h1_template, \
    signal_l1_template, signal_v1_template= get_waveform(mass1,\
                                                         mass2,spin1z,spin2z,\
                                                         declination,right_ascension,
                                                   polarization,coa_phase)


    whiten_noise_strain, psd = whiten_strain(strain)

    rescaled_signal, whiten_signal, whiten_signal_plus_strain, scale, is_rescaled = \
        rescale_signal_and_whiten(signal_h1_template, strain, psd, target_snr)

    signal_h1_resize = signal_padding_zero(signal_h1,32*4096)

    del rescaled_signal, signal_h1, signal_l1, signal_v1, signal_h1_template,signal_l1_template, signal_v1_template
    gc.collect()

    #print('out data_generation')

    return signal_h1_resize, whiten_noise_strain, whiten_signal, whiten_signal_plus_strain, mass1,\
           mass2, target_snr, psd, is_rescaled

# if __name__ == '__main__':
#     get_strain_file_list('data/', gw_event_gps_list())

