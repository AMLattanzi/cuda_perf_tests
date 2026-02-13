#!/bin/bash

# module_restore
# 
# ml PrgEnv-gnu
# ml cuda
# ml binutils
# #ml craype-accel-nvidia90  # does NOT work with PrgEnv-gnu

MPI_INCLUDE=$(cc --cray-print-opts=cflags)
MPI_LINK="-L$MPICH_DIR/lib -lmpi"
CUDA_MPI_LINK="-L$CUDA_HOME/lib64 -lcudart"

# from Jon Rood https://github.com/Exawind/exawind-manager/blob/46b2a8d85c9b5a13f0f5e94d4ce134e937d999f0/repos/spack_repo/exawind/packages/amr_wind/package.py#L37-L43
GTL_FLAGS="$PE_MPICH_GTL_DIR_nvidia90 $PE_MPICH_GTL_LIBS_nvidia90"

nvcc -arch=sm_90 -x cu -std=c++17 -o mpiHostDevice.exe mpiHostDevice.cpp $MPI_INCLUDE $MPI_LINK $GTL_FLAGS $CUDA_MPI_LINK
