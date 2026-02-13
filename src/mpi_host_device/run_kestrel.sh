#!/bin/bash

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU

# Run with two ranks
srun -N 1 -n 2 --gpus-per-node=2 ./mpiHostDevice.exe
