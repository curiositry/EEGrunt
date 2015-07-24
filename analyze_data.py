from EEGrunt import *

# You need to set this stuff!
source = 'openbci'
path = 'data/'
filename = 'EEG_data.csv'
activity  = "SSVEP"

data = load_data(path, filename, source)

EEG = EEGrunt(data, path, filename, source)

for channel in EEG.channels:
    
    EEG.load_channel(channel)

    EEG.default_plot_title = "Channel "+str(EEG.channel)+"\n"+str(EEG.source.title())+" data loaded from "+str(EEG.filename)
    
    print("Processing channel "+ str(EEG.channel))
        
    # Removes OpenBCI DC offset
    EEG.remove_dc_offset()
    
    # Notches 60hz noise (of you're in Europe, switch to 50Hz)
    EEG.notch_mains_interference()

    # Crunches spectrum data and stores as EEGrunt attribute(s) for reuse
    EEG.get_spectrum_data()
    
    # Returns bandpassed data 
    # (uses scipy.signal butterworth filter)
    # EEG.data = EEG.bandpass(start,stop)

    # Make Spectrogram 
    EEG.spectrogram()

    # Line graph of amplitude over time for a give frequency. range
    EEG.plot_band_power(8,12,"Alpha")
    
    # Standard FFT plot
    # (average power over the course of session for a given frequency)
    EEG.plot_spectrum_avg_fft()
        
    # Plot coherence fft (not tested recently...)
    # s1 = bandpass(seginfo["data"][:,1-1], config['band'])
    # s2 = bandpass(seginfo["data"][:,8-1], config['band'])
    # plot_coherence_fft(s1,s2,"1","8")
