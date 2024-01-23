# Generate-data
This is a simple script for generating realistic samples of synthetic gravitational wave data. The data can be used for machine leanrning experiments.

Most of the code is completed by the gravitational wave signal processing team of Jiangxi University of Science and Technology leaded by Cunliang Ma and Sen Wang. The code depends on the PyCBC package which can be installed by 

pip install pycbc -i https://pypi.tuna.tsinghua.edu.cn

The minimum python version required is 3.9.

To generate data, just run the command:  python --configfile XXX.yml


The file XXX.yml is the config file which is in config directory. An example of the config file (config.yml) is shown below

mass1:

min_mass1: 10.0

max_mass1: 80.0

mass2:

min_mass2: 10.0

max_mass2: 80.0

spin1z:

min_spin1z: 0 

max_spin1z: 0.998

spin2z:

min_spin2z: 0 

max_spin2z: 0.998

snr:

snr_min: 7 

snr_max: 20

input directory

input_dir: data

output directory

output_dir: output

The input_dir contains the real data obtained from LIGO-VIRGO-KAGRA collabration which has the hdf5 format and 4k Hz sampling frequency. The generated data is saved in output_dir with the npz format.




