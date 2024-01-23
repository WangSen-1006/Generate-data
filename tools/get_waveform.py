"""
@ File: generate_data.py
@ Author: Cunliang Ma, Wang Sen et al
@ Email: mcl@jxust.edu.cn
@ Brief: This is the main script that generate the data.
@ Copyright(C): 2023 Jiangxi University of Science and Technology. All rights reserved
"""
import gc

import numpy as np
from pycbc.waveform import get_td_waveform
import matplotlib.pyplot as plt
from pycbc.detector import Detector
import matplotlib
from tools.lowpass import butter_lowpass_filter,butter_highpass_filter
matplotlib.use('TkAgg')


def get_waveform(mass1, mass2, spin1z,spin2z, declination, right_ascension, polarization,coa_phase):
    hp, hc = get_td_waveform(approximant='SEOBNRv4',
                             mass1=mass1,
                             mass2=mass2,
                             spin1z=spin1z,
                             spin2z=spin2z,
                             delta_t=1.0 / 4096,
                             f_lower=6,
                             coa_phase=coa_phase,
                             distance=1000)
    det_h1 = Detector('H1')
    det_l1 = Detector('L1')
    det_v1 = Detector('V1')
    signal_h1_low = det_h1.project_wave(hp,hc,right_ascension,declination,polarization)
    signal_l1_low = det_l1.project_wave(hp,hc,right_ascension,declination,polarization)
    signal_v1_low = det_v1.project_wave(hp,hc,right_ascension,declination,polarization)

    low_signal = signal_h1_low
    signal_h1_low.append_zeros(500)
    signal_l1_low.append_zeros(500)
    signal_v1_low.append_zeros(500)
    signal_h1_template = signal_h1_low.highpass_fir(frequency=20, order=500, remove_corrupted=False)
    signal_l1_template = signal_l1_low.highpass_fir(frequency=20, order=500, remove_corrupted=False)
    signal_v1_template = signal_v1_low.highpass_fir(frequency=20, order=500, remove_corrupted=False)

    # plt.plot(low_signal)
    # plt.plot(signal_h1_template)
    # plt.show()
    length=signal_h1_template.shape[0]

    signal_h1_template_return = []
    signal_h1_low_return = []


    if length>4096*32:
        signal_h1_template_return = signal_h1_template[length-4096*32:]
        signal_h1_low_return = signal_h1_low[length-4096*32:]
    else:
        signal_h1_template_return = signal_h1_template
        signal_h1_low_return = signal_h1_low

    # print('len signal_h1 template '+str(signal_h1_template.shape[0]))
    del signal_h1_template, signal_h1_low
    gc.collect()

    return signal_h1_low_return, signal_l1_low, signal_v1_low, signal_h1_template_return, signal_l1_template, signal_v1_template


