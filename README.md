# EEGrunt: A Collection Python EEG Analysis Utilities
<br> Working with EEG (electroencephalography) data is hard, and this little library aims to make it easier. EEGrunt consists of a collection of functions for reading EEG data a CSV files, converting and filtering it in various ways, and finally generating pretty and informative visualizations.

EEGrunt is compatible with data from OpenBCI, but could easily be modified for other EEG aquisition hardware. You could always send me another headset and I add support :)

EEGrunt has bandpass, notch, and highpass filters for cleaning up powerline interference, OpenBCI's DC offset, and zeroing in on the frequency band you want to analyze.

EEGrunt makes it easy to generate signal plots, amplitude trend graphs, spectrograms, and FFT (fast-fouier transform) graphs.

## Getting Started
1. Download or clone the repo
2. Run `sudo bash install_dependencies.sh`
3. Edit `config.py` to match your environment
4. Take a look in `openbci_analysis.py` and edit at will, or create your own script using `EEGrunt_functions.py`
5. Run it: `python openbci_analysis.py`
