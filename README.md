# FDTDAudioCUDA
FDTD Audio Simulation using CUDA with a visualizer written in C + GTK 

## Compilation instructions

This code requires the dev versions of GTK+-3.0 (`sudo apt-get install libgtk-3-dev`) and lbsndfile (`sudo apt-get install libsndfile1-dev`) to compile.

Compile the CUDA simulation code with `nvcc -o sim sim.cu` and the GUI visualizer code with `gcc ``pkg-config gtk+-3.0 --cflags`` visualizer.c -lm -o visualizer ``pkg-config gtk+-3.0 --libs`` `. You can then run the visualizer with `./visualizer`.

You can also optionally choose to run the CUDA simulator from the command line with `./sim [-i infile] [-o outfile] [-t timesteps] [-g gridsize] [-b blockdimensions] [-s sourcecount [[sourcefile sourcepos]...]]`. This allows easy programmatic access to the simulator by your own scripts or programs.

## License information

This code is licensed under the MIT license, included in the LICENSE file

Libsndfile is released under the GNU LGPL, a copy of which is included in this repo
