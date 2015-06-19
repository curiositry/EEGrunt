#!/usr/bin/env python
'''
What's in the Box:
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
- smooth(x,window_len=11,window='hanning')
    - Smooths things so you can tell what's going on with a line plot
- plotit(plt, plotname="")
    - Saves or shows plot, depending on config
- signalplot(data,x_values,x_label,y_label,title)
    - Make a basic signal plot
- convertToFreqDomain(eeg_data_uV, overlap)
- get_spec_psd_per_bin(data)
- spectrogram(data,title,fs_Hz)
    - Make pretty, pretty spectrograms!
- spectrum_avg(spec_PSDperHz,freqs,title)
- plot_amplitude_over_time (x, data, title)
    - Makes a averaged, smoothed trend graph
'''
    
from config import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
# import shelve
# from scipy.signal import butter, lfilter, freqz
from scipy import signal

fs_Hz = config['fs_Hz']
NFFT = config['NFFT']
t_lim_sec = config['t_lim_sec']
overlap  = config['NFFT'] - int(0.25 * config['fs_Hz'])

font = {'family' : 'Ubuntu, Helvetica, Open Sans'}

plt.rc('font', **font)

def packet_check(data):
    data_indices = data[:, 0]
    d_indices = data_indices[2:]-data_indices[1:-1]
    n_jump = np.count_nonzero((d_indices != 1) & (d_indices != -255))
    print("Packet counter discontinuities: " + str(n_jump))
    return n_jump  

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

def bandpass(eeg_data_uV, band):
    bp_Hz = np.zeros(0)
    bp_Hz = np.array([band[0],band[1]])
    b, a = signal.butter(3, bp_Hz/(fs_Hz / 2.0),'bandpass')
    eeg_data_uV = signal.lfilter(b, a, eeg_data_uV, 0)
    print("Bandpass filtering to: " + str(bp_Hz[0]) + "-" + str(bp_Hz[1]) + " Hz")
    return eeg_data_uV
    
def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def avg_samples(arr):
    avgd_data = np.array([])
    for i in range(len(arr)):
        # while x < alpha_max_uVperSqrtBin[:100:5]:
        #     print(i)
        #     print(x)
        #     meen = sum(arr[i-5:i]) / 5
        #     np.append(x,avgd_data)
        #     print avgd_data
        if i % config['sample_block'] == 0:
            # if abs(cutoff_uVperSqrtBin - arr[i]) < 1:
            #     print("Too high!!!!")
            #     print(arr[i])
            #     arr[i] = sum(arr[(i - config['sample_block'])-1:i-1:]) / config['sample_block']
            #     print arr[i]
            #     print ":new"
            meen = sum(arr[i-config['sample_block']:i]) / config['sample_block']
            # if meen > cutoff_uVperSqrtBin:
            #     print("meen over")  
            #     if int(arr[i]) > int(cutoff_uVperSqrtBin):
            #         print "item over:"
            #         print arr[i]
            #         print "end"
            meen = sum(arr[i-config['sample_block']:i]) / config['sample_block']
            avgd_data = np.append(meen, avgd_data)
            # print(avgd_data)
            # while num < 5:
            #     print("while")
            #     avgd_data = np.array([])
            #     idx = i - num
            #     avgd_data[num] = alpha_max_uVperSqrtBin[idx]
            #     print(avgd_data[num-1])
            #     num =  num + 1
            # print("while done")
            # alpha_max_uVperSqrtBin_avgd = []
            # print(sum(avgd_data)/len(avgd_data))
            # alpha_max_uVperSqrtBin_avgd[i] = sum(avgd_data)/len(avgd_data)
    return avgd_data

def plotit(plt, plotname=""):
    if config['plot'] == 'show':
        plt.show()
    if config['plot'] == 'save':
        plt.savefig('plots/EEGrunt '+plotname+'.png')

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
    spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)                 
    return spec_PSDperBin, freqs, t  #convert to "per bin"

def spectrogram(data,title):
    FFTstep = 0.5*fs_Hz  # do a new FFT every half second
    overlap = NFFT - FFTstep  # half-second steps
    f_lim_Hz = [0, 50]   # frequency limits for plotting

    plt.figure(figsize=(10,5))
    data = data - np.mean(data,0)
    ax = plt.subplot(1,1,1)
    spec_PSDperBin, freqs, t = get_spec_psd_per_bin(data)
    
    plt.pcolor(t, freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25,26])
    # plt.xlim(t[0], t[-1]+1)

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
    plotit(plt, 'Channel '+str(config['channel'])+' spectrogram')

    return

def plot_spectrum_avg_fft(spec_PSDperHz,freqs,title):
    spectrum_PSDperHz = np.mean(spec_PSDperHz,1)
    chan_avg = 10*np.log10(spectrum_PSDperHz)
    plt.figure(figsize=(10,5))
    plt.plot(freqs, chan_avg)  # dB re: 1 uV
    plt.xlim((0,60))
    plt.ylim((-30,50))
    plotname = 'Channel '+str(config['channel'])+' Spectrum Average FFT Plot'
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD per Hz (dB re: 1uV^2/Hz)')
    plt.title("Channel "+str(config['channel'])+" Spectrum Average FFT Plot\n"+config["filename"])
    plotit(plt, plotname)
    
def plot_amplitude_over_time (x, data, title):
    title = 'Trend Graph of '+config['band'][2]+' Band EEG Amplitude over Time, Channel '+str(config['channel'])+'\n'+str(config['filename'])
    plt.plot(x, data)
    plt.ylim([-10, 10])
    plt.xlim(len(x)/10)
    plt.xlabel('Time (sec)')
    plt.ylabel('EEG Amplitude (uVrms)')
    plt.title(title)
    plotit(plt, 'Channel '+str(config['channel'])+' trend graph')

def plot_coherence_fft(s1,s2,title, chan_a, chan_b):
    plt.figure()
    plt.ylabel("Coherence")
    plt.xlabel("Frequency (Hz)")
    plt.title("Coherence between channels "+chan_a+" and " +chan_b +" in the "+str(config['band'][0])+"-"+str(config['band'][1])+" band.")
    plt.grid(True)
    plt.xlim(config['band'][0],config['band'][1])
    cxy, f = plt.cohere(s1, s2, NFFT, fs_Hz)
    plotit(plt)