# EEGrunt: A Collection Python EEG Analysis Utilities

#### [[New!] Read the announcement post &raquo;][1]

<br> Working with EEG (electroencephalography) data is hard, and this little library aims to make it easier. EEGrunt consists of a collection of functions for reading EEG data from CSV files, converting and filtering it in various ways, and finally generating pretty and informative visualizations.

EEGrunt is compatible with data from OpenBCI ~~, but could easily be modified for other EEG acquisition hardware. You could always send me another headset and I add support :)~~ and Muse. 

EEGrunt has bandpass, notch, and highpass filters for cleaning up powerline interference, OpenBCI's DC offset, and zeroing in on the frequency band you want to analyze.

EEGrunt makes it easy to generate signal plots, amplitude trend graphs, spectrograms, and FFT (fast-fouier transform) graphs, etc.

~~More documentation on the way!~~  [Read the announcement post for the official tutorial!][1]

## Getting Started
1. Download or clone the repo: `git clone https://github.com/curiositry/EEGrunt`
2. Run `sudo bash install_linux_dependencies.sh` (tell me if this doesn’t work)
4. Take a look in `analyze_data.py` and edit at will, or create your own script using `EEGrunt.py`. **Make sure to set the required variables — device, path, and filename.**
5. Run it: `python analyze_data.py`

[1]: http://www.autodidacts.io/eegrunt-open-source-python-eeg-analysis-and-processing-utilities/
