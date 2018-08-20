# FDTDAudioCUDA
FDTD Audio Simulation using CUDA

## Compilation instructions

This code requires the dev versions of GTK+-3.0 (`sudo apt-get install libgtk-3-dev`) and lbsndfile (`sudo apt-get install libsndfile1-dev`)

Compile the CUDA simulation code with `nvcc -o sim sim.cu`, and optionally run it with `./sim [-i infile] [-o outfile] [-t timesteps] [-g griddimensions] [-b blockdimensions]`

Or you can compile the GUI visualizer code with `gcc ``pkg-config gtk+-3.0 --cflags`` visualizer.c -o visualizer ``pkg-config gtk+-3.0 --libs`` ` and run with `./visualizer`(but you must compile the CUDA code first)

## License information

This code is licensed under the MIT license, included in the LICENSE file

Libsndfile is released under the GNU LGPL, a copy of which is included in this repo
