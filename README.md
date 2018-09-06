# FDTDAudioCUDA
FDTD Audio Simulation using CUDA with a visualizer written in C + GTK 

## Compilation instructions

This code requires the dev versions of GTK+-3.0 (`sudo apt-get install libgtk-3-dev`) and lbsndfile (`sudo apt-get install libsndfile1-dev`) to compile.

Compile the CUDA simulation code with `nvcc -o sim sim.cu` and the GUI visualizer code with `gcc ``pkg-config gtk+-3.0 --cflags`` visualizer.c -lm -o visualizer ``pkg-config gtk+-3.0 --libs`` `.

## GUI interface

Just run the GUI visualizer with ./visualizer.

## Programmatic interface

You can also optionally choose to run the CUDA simulator from the command line, which allows a simple interface for scripts/programs.

The simulator can be passed options in the format `./sim [-i infolder] [-o outfolder] [-t timesteps] [-g gridsize] [-b blockdimensions]`. 

### Audio input

The simulation will run with no audio inputs by default, unless the input folder includes an `audio_ledger.txt` file. This will list all audio inputs files, which must be in the same directory as it. It contains one line, in the format `xpos ypos zpos audiofilename`, per audio file.

### Grid input

The simulation will run on an empty grid by default, unless the input folder includes a `sim_state.bin` file. This contains all values in the simulation at a particular time. By default, the simulator outputs the last frame of each simulation to a `sim_state.bin` file in the output folder. If you want to continue a simulation from where it stopped, you'll need to include this file in your input folder.

## License information

This code is licensed under the MIT license, included in the LICENSE file

Libsndfile is released under the GNU LGPL, a copy of which is included in this repo
