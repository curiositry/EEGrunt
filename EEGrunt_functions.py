import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import shelve

def signalplot(data,x_values,x_label,y_label,title):
    plt.figure(figsize=(10,5))
    plt.subplot(1,1,1)
    plt.plot(x_values,data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()


# Create spectrogram of specific channel
def spectrogram(data,title,fs_Hz):

    NFFT = 256*2  # pick the length of the fft
    FFTstep = 0.5*fs_Hz  # do a new FFT every half second
    overlap = NFFT - FFTstep  # half-second steps
    f_lim_Hz = [2, 50]   # frequency limits for plotting

    plt.figure(figsize=(10,5))

    data = data - np.mean(data,0)

    ax = plt.subplot(1,1,1)

    spec_PSDperHz, freqs, t = mlab.specgram(data,
                               NFFT=NFFT,
                               window=mlab.window_hanning,
                               Fs=fs_Hz,
                               noverlap=overlap
                               ) # returns PSD power per Hz

    spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)  #convert to "per bin"

    plt.pcolor(t, freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    #plt.clim(20-7.5-3.0+np.array([-30, 0]))

    plt.clim([-25,25])
    plt.xlim(t[0], t[-1])

    #plt.xlim(np.array(t_lim_sec)+np.array([-10, 10]))
    #plt.ylim([0, fs_Hz/2.0])  # show the full frequency content of the signal

    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)

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



