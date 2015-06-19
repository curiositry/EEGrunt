from config import *
from EEGrunt_functions import *

# Set shorter vars from config vars (transitional)
fs_Hz     = config['fs_Hz']
NFFT      = config['NFFT'] 
t_lim_sec = config['t_lim_sec']
nchannels = config['nchannels']
channels  = config['channels']
channel   = config['channel']
overlap  = config['NFFT'] - int(0.25 * config['fs_Hz'])
activity  = "meditation"
skiprows = 5
if(config['filename'][(len(config['filename']) - 3):] == "csv"):
    skiprows = 2

# Add more if you want to generate plots for different section of an experiment
data_segments = [
    {
    "start_time":50 ,
    "end_time":100,
    "title":config['filename'],
    "type":activity
    }]
    

# Load up the numpy array
raw_data = np.loadtxt(config['path'] + config['filename'],
	          delimiter=',',
	          skiprows=skiprows,
	          usecols=(0,1,2,3,4,5,6,7,8)
	          )
              
# create a vector with the time of each sample              
t_sec = np.arange(len(raw_data[:, 0])) / fs_Hz

# Get data for segements, if your doing multiple segments
for segment_id in range(len(data_segments)):
    start = data_segments[segment_id]["start_time"]
    end = data_segments[segment_id]["end_time"]
    segment_duration = end - start
    data_segments[segment_id]["data"] = raw_data[(fs_Hz*start):(fs_Hz*end), 1:(nchannels+1)]

# Check packets
packet_check(raw_data)

# do stuff (like make plots) of each chunk of data
for segment_id in range(len(data_segments)):
    # Set data, depending if it's segmented
    seginfo = data_segments[segment_id]
    data = raw_data[:, channel:]
    data = seginfo["data"][:,(channel-1)]
    seg_t = np.arange(len(data)) / fs_Hz

    
    # Set default plot title
    title = "Channel "+str(channel)+"\n"+seginfo["title"]
    
    # Filter data
    unbandpassed_data = remove_dc_offset(data)
    unbandpassed_data = notch_mains_interference(unbandpassed_data)
    data = bandpass(unbandpassed_data, config['band'])

    # Make Spectrogram 
    #spectrogram(data, title)

    # Convert things for trend graph (for trend graph)
    spec_PSDperBin, freqs, t_spec = get_spec_psd_per_bin(data)
    full_spec_PSDperBin, full_t_spec, freqs = convertToFreqDomain(data,  overlap)    
    bool_inds = (freqs > config['band'][0]) & (freqs < config['band'][1])
    band_max_uVperSqrtBin = np.sqrt(np.amax(full_spec_PSDperBin[bool_inds, :], 0))
    avgd_data = avg_samples(band_max_uVperSqrtBin)   
    
    # Make trend graph     
    #plot_amplitude_over_time(full_t_spec[::config['sample_block']], smooth(avgd_data[:len(avgd_data)-10:]), title)
    
    # Convert things for FFT
    # hz_data = data - np.mean(unbandpassed_data,0)
    spec_PSDperHz, freqs2, t2 = mlab.specgram(unbandpassed_data,
                                NFFT=NFFT,
                                window=mlab.window_hanning,
                                Fs=fs_Hz,
                                noverlap=overlap
                               ) 
    # Plot FFT
    #plot_spectrum_avg_fft(spec_PSDperHz,freqs2,title)
    
    # Plot coherence fft
    s1 = bandpass(seginfo["data"][:,1-1], config['band'])
    s2 = bandpass(seginfo["data"][:,8-1], config['band'])
    plot_coherence_fft(s1,s2,title,"1","8")
