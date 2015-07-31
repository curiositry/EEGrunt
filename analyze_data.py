import EEGrunt

# Required settings #

# Data source. Options:
# 'openbci' for data recorded with OBCI GUI;
# 'openbci-openvibe' for OBCI data recorded with OpenViBE's csv writer
# 'muse' for data from Muse headset
source = 'openbci'

# Path to EEG data file
path = 'data/'

# EEG data file name
filename = 'EEG_data.csv'

# Activity label (used in some plots and such)
activity  = "SSVEP"

# Initialize
EEG = EEGrunt(path, filename, source)


# Here we can set some additional properties
# The 'plot' property determines whether plots are displayed or saved.
# Possible values are 'show' and 'save'
EEG.plot = 'show'


# Load the EEG data
EEG.load_data()


for channel in EEG.channels:
    
    EEG.load_channel(channel)

    EEG.default_plot_title = "Channel "+str(EEG.channel)+"\n"+str(EEG.source.title())+" data loaded from "+str(EEG.filename)
    
    print("Processing channel "+ str(EEG.channel))
        
    # Removes OpenBCI DC offset
    EEG.remove_dc_offset()
    
    # Notches 60hz noise (if you're in Europe, switch to 50Hz)
    EEG.notch_mains_interference()

    # Crunches spectrum data and stores as EEGrunt attribute(s) for reuse
    EEG.get_spectrum_data()
    
    # Returns bandpassed data 
    # (uses scipy.signal butterworth filter)
    # EEG.data = EEG.bandpass(start,stop)

    # Make Spectrogram 
    EEG.spectrogram()

    # Line graph of amplitude over time for a given frequency range.
    # Arguments are start frequency, end frequency, and label
    EEG.plot_band_power(8,12,"Alpha")
    
    # Power spectrum
    EEG.plot_spectrum_avg_fft()
        
    # Plot coherence fft (not tested recently...)
    # s1 = bandpass(seginfo["data"][:,1-1], config['band'])
    # s2 = bandpass(seginfo["data"][:,8-1], config['band'])
    # plot_coherence_fft(s1,s2,"1","8")
