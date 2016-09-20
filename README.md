# EEGrunt: A Collection Python EEG (+ECG) Analysis Utilities

### [READ THE ANNOUNCEMENT POST &raquo;][1]

Working with EEG (electroencephalography) data is hard, and this little library aims to make it easier. EEGrunt consists of a collection of functions for reading EEG data from CSV files, converting and filtering it in various ways, and finally generating pretty and informative visualizations.

**Update:** We’ve added functions to plot heart rate and heart rate variability from recorded OpenBCI ECG (electrocardiography) data. You can test these out with the `analyze_ecg_channel.py` and `analyze_ecg_data.py` demo scripts. We’ve posted a new tutorial on our blog to get you started: **[EEGrunt update: Analyze heart rate and HRV with Python][2]**


## Features

1. EEGrunt is compatible with data from OpenBCI and Muse.

2. EEGrunt has bandpass, notch, and highpass filters for cleaning up powerline interference, OpenBCI's DC offset, and zeroing in on the frequency band you want to analyze.

3. EEGrunt makes it easy to generate signal plots, amplitude trend graphs, spectrograms, and FFT (fast-fouier transform) graphs, etc.


## Getting Started

1. Download or clone the repo: `git clone https://github.com/curiositry/EEGrunt`
2. Run `sudo bash install_linux_dependencies.sh` (tell me if this doesn’t work)
3. Take a look in `analyze_data.py` and edit at will, or create your own script using `EEGrunt.py`. **Make sure to set the required variables — device, path, and filename.**
4. Run it: `python analyze_data.py`
5. [Read the announcement post for the official tutorial!][1]
6. [Optional] Interested in [analyzing ECG  data with EEGrunt][2]? Take a look at the [new tutorial][2].

[1]: http://www.autodidacts.io/eegrunt-open-source-python-eeg-analysis-utilities/
[2]: http://www.autodidacts.io/analyze-ecg-heart-rate-and-hrv-with-python-and-eegrunt/
