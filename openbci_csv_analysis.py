#!/usr/bin/env python
# to install dependencies: 
# sudo apt-get install python-pip && pip install matplotlib

from EEGrunt_functions import *

fs_Hz = 250.0        # assumed sample rate for the EEG data

pname = ''
fname = 'EEG_data.csv'
t_lim_sec = [0, 1530.0]

nchannels = 8
channel = 6

def signalplot(data,x_values,x_label,y_label,title):
    plt.figure(figsize=(10,5))
    plt.subplot(1,1,1)
    plt.plot(x_values,data)
    plt.xlabel(x_label)       
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()


# Create spectrogram of specific channel
def spectrogram(data,title):
    
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
    
    plt.clim([-30,30])
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
    plt.show()
    







data_segments = [
    {
    "start_time":30,
    "end_time":330,
    "title":"Control 1",
    "type":"control"
    },
    {
    "start_time":330,
    "end_time":630,
    "title":"Trial 1",
    "type":"trial"
    },
    {
    "start_time":630,
    "end_time":930,
    "title":"Control 2",
    "type":"control"
    },
    {
    "start_time":930,
    "end_time":1230,
    "title":"Trial 2",
    "type":"trial"
    },
    {
    "start_time":30,
    "end_time":1230,
    "title":"Control (5min), Trial (5min), Control (5min), Trial (5min)",
    "type":"combined"
    }]



    
# load data into numpy array

data = np.loadtxt(pname + fname,
                  delimiter=',',
                  skiprows=2,
                  usecols=(0,1,2,3,4,5,6,7,8,9,10)
                  )



# parse the data out of the values read from the text file

#data_segments = shelve.open("eeg_segments",writeback=True)
#print data_segments

#data_segments = 'blah3'

#del data_segments['someother key4']

#data_segments.close()

#exit()

#if(len(data_segments) < 4):
    
#    print "Overwriting data segments..."
    





for segment_id in range(len(data_segments)):
    
    start = data_segments[segment_id]["start_time"]
    end = data_segments[segment_id]["end_time"]
    
    data_segments[segment_id]["data"] = data[(fs_Hz*start):(fs_Hz*end), 1:(nchannels+1)]


#print data_segments

#data_title = "All channel average, full length"
#eeg_data_uV = np.mean(eeg_data_uV,1)
#eeg_data_uV = eeg_data_uV[...,np.newaxis]
#z = np.zeros((len(eeg_data_uV),1))
#np.append(eeg_data_uV, z, axis=1)




# create a vector with the time of each sample
t_sec = np.arange(len(data[:, 0])) / fs_Hz


for segment_id in range(len(data_segments)):
    
    seginfo = data_segments[segment_id]
    data = seginfo["data"][:,(channel-1)]
    title = seginfo["title"] + ", Channel "+ str(channel)
    t_sec = np.arange(len(data)) / fs_Hz
    
    #Create signal plot
    signalplot(data,
               t_sec,
               "Time (seconds)",
               "Microvolts",
               title
               )
    
    #Calculate the spectrum
    spec_PSDperHz, freqs, t = spectrogram(data,title)
    
    data_segments[segment_id]["channels"] = {channel: {
                    "spec_PSDperHz": spec_PSDperHz,
                    "freqs": freqs,
                    "t":t                
                }
            }
    #data_segments[segment_id][channel][] = freqs
    #data_segments[segment_id][channel]["t"] = 
    
    #Calculate spectrum avg
    spectrum_avg(spec_PSDperHz,freqs,title)
    
    


plt.figure(figsize=(10,5))
ax = plt.subplot(1,1,1)
ax.set_color_cycle(['gray', 'orange', 'gray', 'orange'])
plt.xlim((0,50))
plt.ylim((-20,20))

for segment_id in range(len(data_segments)):
    
    seginfo = data_segments[segment_id]
    data = seginfo["data"][:,(channel-1)]
    title = seginfo["title"] + ", channel"+ str(channel)
    
    #print seginfo["channels"]
  
    spectrum_PSDperHz = np.mean(seginfo["channels"][channel]["spec_PSDperHz"],1)
    
    seg_chan_avg = 10*np.log10(spectrum_PSDperHz)
    
    data_segments[segment_id]["channels"][channel]["seg_chan_avg"] = seg_chan_avg 
    
    ax.plot(seginfo["channels"][channel]["freqs"],seg_chan_avg)  # dB re: 1 uV
    
#     print seginfo["channels"]
    

plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD per Hz (dB re: 1uV^2/Hz)')
plt.title("All segments avg frequencies (Channel "+ str(channel)+")")
plt.show()    

spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)
spec_PSDperBin[1,1]

