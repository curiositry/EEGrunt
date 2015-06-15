#!/usr/bin/env python
'''
What’s in the Box:
==================
- Import: config, pyplot, numpy, mlab, shelve, scipy utils
- packet_check(data)
    - Checks for packet counter discontinuities and prints result
- remove_dc_offset(eeg_data_uV)
    - Highpass filter to remove OpenBCI DC offset
- notch_mains_interference(eeg_data_uV)
    - Notch filter to remove 60hz mains interference + harmonics
- bandpass(eeg_data_uV, lowcut, highcut)
    - Generic butterworth bandpass to narrow down frequency bands
- signalplot(data,x_values,x_label,y_label,title)
    - Make a basic signal plot
- convertToFreqDomain(eeg_data_uV, overlap)
- get_spec_psd_per_bin(data)
- spectrogram(data,title,fs_Hz)
    - Make pretty, pretty spectrograms!
- spectrum_avg(spec_PSDperHz,freqs,title)
'''
    
from config import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import shelve
from scipy.signal import butter, lfilter, freqz
from scipy import signal

fs_Hz = config['fs_Hz']
NFFT = config['NFFT']

def packet_check(data):
    data_indices = data[:, 0]
    d_indices = data_indices[2:]-data_indices[1:-1]
    n_jump = np.count_nonzero((d_indices != 1) & (d_indices != -255))
    return "Packet counter discontinuities: " + str(n_jump)              

def remove_dc_offset(eeg_data_uV):
    hp_cutoff_Hz = 1.0
    b, a = signal.butter(2, hp_cutoff_Hz/(fs_Hz / 2.0), 'highpass') 
    eeg_data_uV = signal.lfilter(b, a, eeg_data_uV, 0)
    print("Highpass filtering at: " + str(hp_cutoff_Hz) + " Hz")
    return eeg_data_uV
    
def notch_mains_interference(eeg_data_uV):
    notch_freq_Hz = np.array([60.0, 120.0])  # main + harmonic frequencies
    for freq_Hz in np.nditer(notch_freq_Hz):  # loop over each target freq
        bp_stop_Hz = freq_Hz + 3.0*np.array([-1, 1])  # set the stop band
        b, a = signal.butter(3, bp_stop_Hz/(fs_Hz / 2.0), 'bandstop') 
        eeg_data_uV = signal.lfilter(b, a, eeg_data_uV, 0)
        print("Notch filter removing: " + str(bp_stop_Hz[0]) + "-" + str(bp_stop_Hz[1]) + " Hz")
        return eeg_data_uV

def bandpass(eeg_data_uV, lowcut, highcut):
    bp_Hz = np.zeros(0)
    bp_Hz = np.array([lowcut, highcut])
    b, a = signal.butter(3, bp_Hz/(fs_Hz / 2.0),'bandpass')
    eeg_data_uV = signal.lfilter(b, a, eeg_data_uV, 0)
    print("Bandpass filtering to: " + str(bp_Hz[0]) + "-" + str(bp_Hz[1]) + " Hz")
    return eeg_data_uV


def signalplot(data,x_values,x_label,y_label,title):
    plt.figure(figsize=(10,5))
    plt.subplot(1,1,1)
    plt.plot(x_values,data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Plot of "+title)
    plt.tight_layout()
    
def convertToFreqDomain(eeg_data_uV, overlap):
    spec_PSDperHz, freqs, t_spec = mlab.specgram(np.squeeze(eeg_data_uV),
                                   NFFT=NFFT,
                                   window=mlab.window_hanning,
                                   Fs=fs_Hz,
                                   noverlap=overlap
                                   ) # returns PSD power per Hz
    # convert the units of the spectral data
    spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)  # convert to "per bin"
    return spec_PSDperBin, t_spec, freqs

def get_spec_psd_per_bin(data):
    overlap = NFFT - int(0.25 * fs_Hz) # quarter-second steps
    spec_PSDperHz, freqs, t = mlab.specgram(data,
                           NFFT=NFFT,
                           window=mlab.window_hanning,
                           Fs=fs_Hz,
                           noverlap=overlap
                           ) # returns PSD power per Hz

    return [spec_PSDperHz * fs_Hz / float(NFFT), freqs, t]  #convert to "per bin"


# Create spectrogram of specific channel
def spectrogram(data,title):
    FFTstep = 0.5*fs_Hz  # do a new FFT every half second
    overlap = NFFT - FFTstep  # half-second steps
    f_lim_Hz = [0, 100]   # frequency limits for plotting

    plt.figure(figsize=(10,5))
    data = data - np.mean(data,0)
    ax = plt.subplot(1,1,1)
    spec_PSDperBin = get_spec_psd_per_bin(data,NFFT)
    t = spec_PSDperBin[1]
    freqs = spec_PSDperBin[2]
    spec_PSDperBin =  spec_PSDperBin[0]
    
    plt.pcolor(t, freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    #plt.clim(20-7.5-3.0+np.array([-30, 0]))
    plt.clim([-25,25])
    plt.xlim(t[0], t[-1])

    #plt.xlim(np.array(t_lim_sec)+np.array([-10, 10]))
    #plt.ylim([0, fs_Hz/2.0])  # show the full frequency content of the signal

    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title("Spectrogram of "+title)

    # add annotation for FFT Parameters
    ax.text(0.025, 0.95,
        "NFFT = " + str(NFFT) + "\nfs = " + str(int(fs_Hz)) + " Hz",
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        backgroundcolor='w')
    #plt.colorbar()
    plt.show()

    return spec_PSDperHz, freqs, t


def spectrum_avg(spec_PSDperHz,freqs,title):
    #find spectrum slices within the time period of interest
    #ind = ((t > t_lim_sec[0]) & (t < t_lim_sec[1]))
    spectrum_PSDperHz = np.mean(spec_PSDperHz,1)
    plt.figure(figsize=(10,5))
    ax = plt.subplot(1,1,1)

    ax.set_color_cycle(['m', 'c', 'm', 'c'])

    ax.plot(freqs, 10*np.log10(spectrum_PSDperHz))  # dB re: 1 uV

    plt.xlim(0, 50)
    plt.ylim(-20, 20)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD per Hz (dB re: 1uV^2/Hz)')
    plt.title(title)
    # plt.show()

