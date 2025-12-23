#!/bin/bash

# Use the mpi built with cuda support
/usr/openmpi-cuda/bin/mpirun -np 2 ./mpiHostDevice.exe
