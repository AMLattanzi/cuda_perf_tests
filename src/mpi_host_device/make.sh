#!/bin/bash

MPI_INCLUDE=$(mpic++ --showme:compile)
MPI_LINK=$(mpic++ --showme:link)
CUDA_MPI_LINK="-L/usr/local/cuda-12.6/lib64 -lcudart"
nvcc -arch=sm_61 -x cu -std=c++17 --expt-extended-lambda \
     $MPI_INCLUDE $MPI_LINK $CUDA_MPI_LINK mpiHostDevice.cpp -o mpiHostDevice.exe

