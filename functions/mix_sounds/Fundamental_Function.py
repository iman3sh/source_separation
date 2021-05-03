# ======================================================================================
# creator iman shahriari
# ======================================================================================

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import random


# In[fundamental function]
def my_wave_read(addr):
    try:
        (wav, sr) = sf.read(addr,dtype='int16')
        return wav, sr
    except:
        print('cannot open wave file!')
        return ERROR_code_WAV_FILE_PROBLEM, None

def my_wave_write(data, addr, sr=16000):
    if data.dtype!='int16':
        print('input type is not int16')
    try:
        sf.write(addr, data, samplerate=sr)
    except:
        print('problem with saving the wave!')

#________________________________________________________
def load_data_frame(addr, mode=None):
    if mode == 'zip':
        df = pd.read_pickle(addr, compression='zip')
    else:
        df = pd.read_pickle(addr)
    return df

def save_data_frame(df, addr, mode=None):
    if mode == 'zip':
        df.to_pickle(addr, compression='zip')
    else:
        df.to_pickle(addr)

def data_frame_append(df, data):
    df = df.append(data, ignore_index=True)
    return df

#________________________________________________________
def polar2z(r,theta):
    return r * np.exp( 1j * theta )

def z2polar(z):
    return np.abs(z), np.angle(z)

#________________________________________________________
def strided_axis0(a, L):
    a = a
    # INPUTS :
    # a is array
    # L is length of array along axis=0 to be cut for forming each subarray

    # Length of 3D output array along its axis=0
    nd0 = a.shape[0] - L + 1

    # Store shape and strides info
    m,n = a.shape
    s0,s1 = a.strides

    # Finally use strides to get the 3D array view
    return np.lib.stride_tricks.as_strided(a, shape=(nd0,L,n), strides=(s0,s0,s1))

def singleframe2context(input, context_size):
    noisy_fft_abs_pad = np.pad(input, ((context_size, context_size), (0, 0)), 'wrap')
    noisy_fft_abs_context = strided_axis0(noisy_fft_abs_pad, int(context_size * 2 + 1))
    return noisy_fft_abs_context

def context2singleframe(input):
    context_size = int(len(input[0])/2)
    output = []
    for i in range(len(input)):
        output.append(input[i][context_size])
    return np.asarray(output)

def plot_spectrogram(abs_fft):
    plt.pcolormesh(abs_fft.T, shading='gouraud',cmap='hot')
    plt.ylabel('Frequency bin [fourier coefficients number]')
    plt.xlabel('Time [sec]')
    plt.show()

#__________________________ make noisy file ______________________________
def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms

def add_noise(clean_amp, noise_amp, SNR):
    if clean_amp.dtype != 'float64':
        clean_amp = clean_amp.astype(np.float64)
    if noise_amp.dtype != 'float64':
        noise_amp = noise_amp.astype(np.float64)
    clean_rms = np.sqrt(np.mean(np.square(clean_amp), axis=-1))

    start = random.randint(0, len(noise_amp) - len(clean_amp))
    divided_noise_amp = noise_amp[start: start + len(clean_amp)]
    noise_rms = np.sqrt(np.mean(np.square(divided_noise_amp), axis=-1))

    adjusted_noise_rms = cal_adjusted_rms(clean_rms, SNR)

    adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms)
    noisy_amp = (clean_amp + adjusted_noise_amp)
    # Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if noisy_amp.max(axis=0) > max_int16 or noisy_amp.min(axis=0) < min_int16:
        if noisy_amp.max(axis=0) >= abs(noisy_amp.min(axis=0)):
            reduction_rate = max_int16 / noisy_amp.max(axis=0)
        else:
            reduction_rate = min_int16 / noisy_amp.min(axis=0)
        noisy_amp = noisy_amp * (reduction_rate)
        clean_amp = clean_amp * (reduction_rate)
    clean_amp = clean_amp.astype(np.int16)
    noisy_amp = noisy_amp.astype(np.int16)
    # plt.subplot(211)
    # plt.plot(noisy_amp)
    # plt.subplot(212)
    # plt.plot(clean_amp)
    # plt.show()
    return  clean_amp, noisy_amp