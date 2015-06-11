from EEGrunt_functions import *

fs_Hz = 250.0        # assumed sample rate for the EEG data

pname = ''
fname = 'EEG_data.txt'
t_lim_sec = [0, 1530.0]

nchannels = 8
channels = [1, 2, 3,4,5,6,7,8]


data_segments = [
    {
    "start_time":30 ,
    "end_time":999999999999,
    "title":"Meditation",
    "type":"meditation"
    }]



for channel in channels:
	# load data into numpy array

	data = np.loadtxt(pname + fname,
		          delimiter=',',
		          skiprows=5,
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
	    title = seginfo["title"] + ", channel "+ str(channel)
	    t_sec = np.arange(len(data)) / fs_Hz

	    #Create signal plot

	    # signalplot(data,
	    #            t_sec,
	    #            "Time (seconds)",
	    #            "Microvolts",
	    #            title
	    #            )

	    #Calculate the spectrum
	    spec_PSDperHz, freqs, t = spectrogram(data,title, fs_Hz)

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


# plt.figure(figsize=(10,5))
# ax = plt.subplot(1,1,1)
# ax.set_color_cycle(['gray', 'orange', 'gray', 'orange'])
# plt.xlim((0,50))
# plt.ylim((-20,20))
#
# for segment_id in range(len(data_segments)):
#
#     seginfo = data_segments[segment_id]
#     data = seginfo["data"][:,(channel-1)]
#     title = seginfo["title"] + ", channel"+ str(channel)
#
#     #print seginfo["channels"]
#
#     spectrum_PSDperHz = np.mean(seginfo["channels"][channel]["spec_PSDperHz"],1)
#
#     seg_chan_avg = 10*np.log10(spectrum_PSDperHz)
#
#     data_segments[segment_id]["channels"][channel]["seg_chan_avg"] = seg_chan_avg
#
#     ax.plot(seginfo["channels"][channel]["freqs"],seg_chan_avg)  # dB re: 1 uV
#
# #     print seginfo["channels"]
#
#
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('PSD per Hz (dB re: 1uV^2/Hz)')
# plt.title("All segments avg frequencies (Channel "+ str(channel)+")")
# plt.show()

# spec_PSDperBin[1,1]
