# Use 25 Hz as the common sampling rate

All data is resampled to 25 Hz in the canonical format, although several datasets are recorded at 50–238 Hz. 25 Hz is the only rate all four target vendors can supply: the Samsung Health Sensor SDK caps accelerometer streaming at 25 Hz and some Garmin devices report 25 Hz, and a lower rate also keeps sensor-streaming battery cost down. The cost is that the short impact spike of a fall is under-represented; models are trained on anti-aliased 25 Hz data and never on higher-rate signals downsampled naively.
