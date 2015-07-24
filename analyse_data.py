from EEGrunt import *

# You need to set this stuff!
source = 'openbci'
path = 'data/'
filename = 'converted_OpenBCI-RAW-2015-04-23_08-46-58-rf-meditation2.csv'
activity  = "SSVEP"

data = load_data(path, filename, source)

grnt = EEGrunt(data, path, filename, source)

# TODO Integrate data segmentation as a EEGrunt method
# data_segments = grnt.data_segments

# for segment_id in range(len(data_segments)):
#     start = data_segments[segment_id]["start_time"]
#     end = data_segments[segment_id]["end_time"]
#     segment_duration = end - start
#     
#     print("Seg start time")
#     print(start)
#     
#     data_segments[segment_id]["data"] = grnt.raw_data[(grnt.fs_Hz*start):(grnt.fs_Hz*end),:]
#     print(len(grnt.raw_data[:, 0:(grnt.nchannels+1)]))
#     print(data_segments[segment_id]["data"])
    # do stuff (like make plots) of each chunk of data
    # for segment_id in range(len(data_segments)):
    #     # Set data, depending if it's segmented
    #     seginfo = data_segments[segment_id]
    #     # data = raw_data[:, channel:]
    #     grnt.data = seginfo["data"][:,(channel-1)]
    #     
    #     grnt.title = "Channel "+str(channel)+"\n"+seginfo["title"]

grnt.data_all_channels = grnt.data

for channel in grnt.channels:
        
    grnt.channel = channel
    
    grnt.default_plot_title = "Channel "+str(grnt.channel)+"\n"+str(grnt.source.title())+" data loaded from "+str(grnt.filename)
    
    grnt.data = grnt.data_all_channels[:,(grnt.channel+grnt.col_offset)]
        
    print("Processing channel "+ str(grnt.channel))
        
    # Removes OpenBCI DC offset
    grnt.remove_dc_offset()
    
    # Notches 60hz noise (of you're in Europe, switch to 50Hz)
    grnt.notch_mains_interference()

    # Crunches spectrum data and stores as EEGrunt attribute(s) for reuse
    grnt.get_spectrum_data()
    
    # Returns bandpassed data 
    # (uses scipy.signal butterworth filter)
    # grnt.data = grnt.bandpass(start,stop)

    # Make Spectrogram 
    grnt.spectrogram()

    # Line graph of amplitude over time for a give frequency. range
    grnt.plot_band_power(8,12,"Alpha")
    
    # Standard FFT plot
    # (average power over the course of session for a given frequency)
    grnt.plot_spectrum_avg_fft()
        
    # Plot coherence fft (not tested recently...)
    # s1 = bandpass(seginfo["data"][:,1-1], config['band'])
    # s2 = bandpass(seginfo["data"][:,8-1], config['band'])
    # plot_coherence_fft(s1,s2,title,"1","8")
