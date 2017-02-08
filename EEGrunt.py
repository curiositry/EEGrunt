#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from scipy import signal

class EEGrunt:
    def __init__(self, path, filename, source, title = ""):

        self.path = path
        self.filename = filename
        self.source = source
        if(title):
            self.session_title = title
        else:
            self.session_title = source.title()+" data loaded from "+filename

        if self.source == 'openbci' or self.source == 'openbci-openvibe':
            self.fs_Hz = 250.0
            self.NFFT = 256*2
            self.nchannels = 8
            self.channels = [1,2,3,4,5,6,7,8]
            self.col_offset = 0


        if self.source == 'muse':
            self.fs_Hz = 220.0
            self.NFFT = 220*2
            self.nchannels = 4
            self.channels = [1,2,3,4]
            self.col_offset = -1


        self.sample_block = 11

        self.plot = 'show'

        self.overlap  = self.NFFT - int(0.25 * self.fs_Hz)

        self.ecg_threshold_factor = 6
        self.hrv_window_length = 10




    def load_data(self):

        path = self.path
        filename = self.filename
        source = self.source

        print("Loading EEG data: "+path+filename)

        try:
            with open(path+filename) as file:
                pass
        except IOError:
            print 'EEG data file not found.'
            exit()

        if source == 'muse':
            skiprows = 0
            raw_data = []
            with open(path + filename, 'rb') as csvfile:
                for row in csvfile:
                    cols = row.split(',')
                    if(cols[1].strip() == "/muse/eeg"):
                        raw_data.append(cols[2:6])

            dt = np.dtype('Float64')
            raw_data = np.array(raw_data, dtype=dt)

        if source == 'openbci':
            skiprows = 5
            raw_data = np.loadtxt(path + filename,
                          delimiter=',',
                          skiprows=skiprows,
                          usecols=(0,1,2,3,4,5,6,7,8)
                          )


        if source == 'openbci-openvibe':
            skiprows = 1
            raw_data = np.loadtxt(path + filename,
                          delimiter=',',
                          skiprows=skiprows,
                          usecols=(0,1,2,3,4,5,6,7,8)
                          )


        self.raw_data = raw_data

        self.t_sec = np.arange(len(self.raw_data[:, 0])) /self.fs_Hz

        print "Session length (seconds): "+str(len(self.t_sec)/self.fs_Hz)
        print "t_sec last: "+str(self.t_sec[:-1])



    def load_channel(self,channel):
        print("Loading channel: "+str(channel))
        channel_data = self.raw_data[:,(channel+self.col_offset)]
        self.channel = channel
        self.data = channel_data

    def trim_data(self, start, end):
        # Trim data off the beginning and end to get rid of unwanted
        # artifacts (such as from applying and removing electrodes).
        #
        # Arguments 'start' and 'end' are how many seconds to trim
        # from the start and end of the data.
        #
        # Note: this must be applied to a single channel,
        # not to data that has multiple channels. For best results, run it
        # after EEG.notch_mains_interference().

        trim_start_samples = int(start * self.fs_Hz)
        trim_end_samples = int(end * self.fs_Hz)*-1

        if(trim_end_samples == 0):
            trim_end_samples = len(self.data)

        self.data = self.data[trim_start_samples:trim_end_samples]
        self.t_sec = self.t_sec[trim_start_samples:trim_end_samples]


    def packet_check(self):
        data_indices = self.data[:, 0]
        d_indices = data_indices[2:]-data_indices[1:-1]
        n_jump = np.count_nonzero((d_indices != 1) & (d_indices != -255))
        print("Packet counter discontinuities: " + str(n_jump))
        self.n_jump  = n_jump

    def remove_dc_offset(self):
        hp_cutoff_Hz = 1.0

        print("Highpass filtering at: " + str(hp_cutoff_Hz) + " Hz")

        b, a = signal.butter(2, hp_cutoff_Hz/(self.fs_Hz / 2.0), 'highpass')
        self.data = signal.lfilter(b, a, self.data, 0)


    def notch_mains_interference(self):
        notch_freq_Hz = np.array([60.0])  # main + harmonic frequencies
        for freq_Hz in np.nditer(notch_freq_Hz):  # loop over each target freq
            bp_stop_Hz = freq_Hz + 3.0*np.array([-1, 1])  # set the stop band
            b, a = signal.butter(3, bp_stop_Hz/(self.fs_Hz / 2.0), 'bandstop')
            self.data = signal.lfilter(b, a, self.data, 0)
            print("Notch filter removing: " + str(bp_stop_Hz[0]) + "-" + str(bp_stop_Hz[1]) + " Hz")

    def bandpass(self,start,stop):
        bp_Hz = np.zeros(0)
        bp_Hz = np.array([start,stop])
        b, a = signal.butter(3, bp_Hz/(self.fs_Hz / 2.0),'bandpass')
        print("Bandpass filtering to: " + str(bp_Hz[0]) + "-" + str(bp_Hz[1]) + " Hz")
        return signal.lfilter(b, a, self.data, 0)

    # Convenient smoothing function from SciPy cookbook: http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    def smooth(self,x,window_len=11,window='hanning'):
        if x.ndim != 1:
            raise ValueError, "Smooth only accepts 1 dimension arrays."
        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."
        if window_len<3:
            return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Invalid window type in smooth(). Must be one of 'flat', 'hanning', 'hamming', 'bartlett', or 'blackman'"
        s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='valid')
        return y

    def plotit(self, plt, filename=""):
        if self.plot == 'show':
            plt.draw()
        if self.plot == 'save':
            plt.savefig(filename)
            plt.close()

    def showplots(self):
        if self.plot == 'show':
            print("Computation complete! Showing generated plots...")
            plt.show()

    def signalplot(self):
        print("Generating signal plot...")
        plt.figure(figsize=(10,5))
        plt.subplot(1,1,1)
        plt.plot(self.t_sec,self.data)
        plt.xlabel('Time (sec)')
        plt.ylabel('Power (uV)')
        plt.title(self.plot_title('Signal'))
        self.plotit(plt)

    def get_spectrum_data(self):
        print("Calculating spectrum data...")
        self.spec_PSDperHz, self.spec_freqs, self.spec_t = mlab.specgram(np.squeeze(self.data),
                                       NFFT=self.NFFT,
                                       window=mlab.window_hanning,
                                       Fs=self.fs_Hz,
                                       noverlap=self.overlap
                                       ) # returns PSD power per Hz
        # convert the units of the spectral data
        self.spec_PSDperBin = self.spec_PSDperHz * self.fs_Hz / float(self.NFFT)


    def spectrogram(self):
        print("Generating spectrogram...")
        f_lim_Hz = [0, 50]   # frequency limits for plotting
        plt.figure(figsize=(10,5))
        ax = plt.subplot(1,1,1)
        plt.pcolor(self.spec_t, self.spec_freqs, 10*np.log10(self.spec_PSDperBin))  # dB re: 1 uV
        plt.clim([-25,26])
        plt.xlim(self.spec_t[0], self.spec_t[-1]+1)
        plt.ylim(f_lim_Hz)
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.title(self.plot_title('Spectrogram'))
        # add annotation for FFT Parameters
        ax.text(0.025, 0.95,
            "NFFT = " + str(self.NFFT) + "\nfs = " + str(int(self.fs_Hz)) + " Hz",
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            backgroundcolor='w')
        self.plotit(plt, self.plot_filename('Spectrogram'))

    def plot_title(self, title = ""):
        return 'Channel '+str(self.channel)+' '+title+'\n'+self.session_title

    def plot_filename(self,title = ""):
        fn = self.session_title+' Channel '+str(self.channel)+' '+title
        valid_chars = '-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        filename = 'plots/'+(''.join(c for c in fn if c in valid_chars)).replace(' ','_')+'.png'
        return filename


    def plot_spectrum_avg_fft(self):

        print("Generating power spectrum plot")

        spectrum_PSDperHz = np.mean(self.spec_PSDperHz,1)
        plt.figure(figsize=(10,5))
        plt.plot(self.spec_freqs, 10*np.log10(spectrum_PSDperHz))  # dB re: 1 uV
        plt.xlim((0,60))
        plt.ylim((-30,50))
        plotname = 'Channel '+str(self.channel)+' Spectrum Average FFT Plot'
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD per Hz (dB re: 1uV^2/Hz)')

        plt.title(self.plot_title("Power Spectrum"))
        self.plotit(plt, self.plot_filename("Power Spectrum"))


    def plot_band_power(self,start_freq,end_freq,band_name):
        print("Plotting band power over time. Frequency range: "+str(start_freq)+" - "+str(end_freq))
        bool_inds = (self.spec_freqs > start_freq) & (self.spec_freqs < end_freq)
        band_power = np.sqrt(np.amax(self.spec_PSDperBin[bool_inds, :], 0))
        plt.figure(figsize=(10,5))
        plt.plot(self.spec_t,band_power)
        plt.ylim([np.amin(band_power), np.amax(band_power)+1])
        # plt.xlim(len(x)/config['sample_block'])
        plt.xlabel('Time (sec)')
        plt.ylabel('EEG Amplitude (uVrms)')
        plt.title(self.plot_title('Trend Graph of '+band_name+' EEG Amplitude over Time'))
        self.plotit(plt, self.plot_filename(band_name+' EEG Amplitude Over Time'))

    def get_rr_intervals(self):
        print("Getting R-R Interval values...")

        sig1 = self.data

        print("Smoothing data...")

        sig1 = self.smooth(sig1)
        # Lather, rinse, repeat
        sig1 = self.smooth(sig1)

        sig1 = sig1[10:-10] # Smoothing makes the signal longer, so we need to chop it off

        self.signal_diff = np.diff(sig1)
        self.signal_diff = np.append(self.signal_diff,0) # Cheap way to get shape to match...

        abs_diff = np.sqrt(self.signal_diff**2)
        self.ecg_threshold = np.average(abs_diff)*self.ecg_threshold_factor

        print("Threshold: " + str(self.ecg_threshold))

        count = 0
        last_val = .0
        current_rr = .0
        # This array gets a value added for every sample, so can be plotted in the time domain
        self.rr_intervals_array = []
        # This array is just RR values, useful for statistical purposes
        self.rr_intervals_not_indexed_to_samples = []

        for val in self.signal_diff:
            count = count + 1

            if (val > self.ecg_threshold and last_val < self.ecg_threshold):
                current_rr = (count/self.fs_Hz)
                self.rr_intervals_not_indexed_to_samples.append(count)
                count = 0
            last_val = val
            self.rr_intervals_array.append(current_rr)

        self.data = sig1

    def plot_rr_intervals(self):
        if hasattr(self, "rr_intervals_array") == False:
            self.get_rr_intervals()

        print("Plotting ECG signal + R-R intervals...")

        #plt.figure(figsize=(10,5))

        fig, ax1 = plt.subplots()

        ax1.plot(self.t_sec,self.data, label='Smoothed ECG signal')
        ax1.plot(self.t_sec,self.signal_diff, label='Signal diff')
        ax1.plot(self.t_sec,self.data*0+self.ecg_threshold, 'orange', label='Threshold')

        ax1.set_ylabel("Signal power (uV)")
        ax1.legend(loc=3)

        ax2 = ax1.twinx()
        ax2.plot(self.t_sec,self.rr_intervals_array,'r',label='RR intervals')
        ax2.set_ylabel("RR intervals (seconds)")

        ax2.legend(loc=4)

        plt.xlabel('Time (sec)')
        plt.title(self.plot_title('ECG signal processing steps'))

        plt.autoscale(True,'both',True)
        self.plotit(plt)

    def plot_heart_rate(self):
        if hasattr(self, "rr_intervals_array") == False:
            self.get_rr_intervals()

        heart_rate_array = []

        err_count = 0
        for val in self.rr_intervals_array:
            # This is probably a heart-beat
            if val > 0.1905:
                heart_rate = 60.0 / val

            # if RR-interval < .1905 seconds, heart-rate > highest recorded value, 315 BPM. Probably an error!
            elif val > 0 and val < 0.1905:
                # So we'll warn the user that the data seems to have issues

                err_count += 1

                # ... and use the mean heart-rate from the data so far:
                if len(heart_rate_array) > 0:
                    heart_rate = np.mean(heart_rate_array)
                else:
                    heart_rate = 60.0
            # Get around divide by 0 error
            else:
                heart_rate = 0.0

            # Append the heart-rate
            heart_rate_array.append(heart_rate)


        if err_count > 0:
            print("WARNING! RR-interval was shorter than fastest recorded heart-beat. [" + str(err_count) + " x]")

        # Get the average heart rate over the session (for the plot title)
        self.avg_heart_rate = np.mean(heart_rate_array)

        plt.figure(figsize=(10,5))
        plt.subplot(1,1,1)
        plt.plot(self.t_sec, heart_rate_array)
        plt.subplot(1,1,1)

        plt.xlabel('Time (sec)')
        plt.ylabel('Heart-rate (BPM)')
        plt.title(self.plot_title('ECG Signal. \n Avg heart-rate: ' + str(int(self.avg_heart_rate)) + " BPM."))
        #plt.ylim(-1, 200)
        plt.autoscale(True,'both',True)
        self.plotit(plt)

    def plot_hrv(self):
        if hasattr(self, "rr_intervals_array") == False:
            self.get_rr_intervals()

        hrv_std_array = []
        index = 1
        err_count = 0
        chunk = []

        # For using time indexed, padded RR data
        '''
        arr = self.rr_intervals_array
        window_length = 20
        window_length_samples = int(window_length*self.fs_Hz)
        x_label = "Samples"
        '''

        # Non-time-indexed unpadded RR data
        arr = self.rr_intervals_not_indexed_to_samples
        window_length = self.hrv_window_length
        window_length_samples = int(window_length*(self.avg_heart_rate/60))
        x_label = "Heart beats"

        print("Data length (samples):"+str(len(arr)))
        print("Window length (samples):"+str(window_length_samples))

        for val in arr:
            if index < int(window_length_samples):
                chunk = arr[:index:]
            else:
                chunk = arr[(index-window_length_samples):index:]

            hrv_std_value = np.std(chunk)
            hrv_std_array.append(hrv_std_value)
            index += 1

        dt = np.dtype('Float64')
        hrv_std_array = np.array(hrv_std_array, dtype=dt)


        self.session_hrv = np.std(self.rr_intervals_not_indexed_to_samples)

        plt.figure(figsize=(10,5))
        plt.subplot(1,1,1)
        plt.plot(hrv_std_array)

        plt.xlabel(x_label)
        plt.ylabel('Standard deviation of R-R intervals (over '+str(self.hrv_window_length)+'s window)')
        plt.title(self.plot_title('ECG Signal. \n  Standard deviation of R-R intervals over session (HRV): ' + str(self.session_hrv)))
        plt.autoscale(True,'both',True)
        self.plotit(plt)

    def plot_coherence_fft(self, s1, s2, chan_a, chan_b):
        plt.figure()
        plt.ylabel("Coherence")
        plt.xlabel("Frequency (Hz)")
        plt.title(self.plot_title("Coherence between channels "+chan_a+" and " +chan_b +" in the "+str(config['band'][0])+"-"+str(config['band'][1])+" Hz band"))
        plt.grid(True)
        plt.xlim(config['band'][0],config['band'][1])
        cxy, f = plt.cohere(s1, s2, NFFT, fs_Hz)
        self.plotit(plt)
