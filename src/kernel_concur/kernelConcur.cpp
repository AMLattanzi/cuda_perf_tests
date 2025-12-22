#include <chrono>
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

#define TILEX 32
#define TILEY 32

// data layout is ptr(row,col)
__global__
void
multByTwo (int nrow,
           int ncol,
           double *data)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if (col < ncol && row < nrow) {
    data[ncol*row + col] *= 2.0f;
  }
}


int
main ()
{
  // Number of streams
  const int nstream = 2;
  cudaStream_t streams[nstream];
  
  // Matrix size on stream
  int nrows  = 1<<12;
  int ncols  = 1<<12;
  int ncells = nrows * ncols;
  auto sizes = ncells * sizeof(double);

  // Matrix size total
  int nrow   = nrows*nstream;
  int ncol   = ncols*nstream;
  int ncell  = ncells * nstream;
  auto size  = ncell  * sizeof(double);

  // Pinned host memory
  double *h_mat;
  cudaMallocHost((void**)&h_mat, size);

  // Initialize host data
  for (int idx(0); idx < ncell; ++idx) {
    h_mat[idx] = (double)idx;
  }

  // Device data w/out streams
  double* d_mat;
  cudaMalloc((void**)&d_mat, size);

  // Start timer
  cudaDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();
  
  // NO STREAMS: Copy H2D, kernel, copy D2H
  //==============================================================
  cudaMemcpyAsync(d_mat, h_mat, size, cudaMemcpyHostToDevice);
  dim3 block(TILEX,TILEY);
  dim3 grid((ncol + TILEX - 1)/TILEX, (nrow + TILEY - 1)/TILEY);
  multByTwo<<<grid, block>>>(nrow, ncol, d_mat);
  cudaMemcpyAsync(h_mat, d_mat, size, cudaMemcpyDeviceToHost);

  // Sync device and host
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "No concurrency compute time (ms): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << std::endl;
  
  // Clean up
  cudaFree(d_mat);

  // Create non-blocking streams and allocate on device
  double *d_mats[nstream];
  for (int istream(0); istream < nstream; ++istream) {
    cudaStreamCreateWithFlags(&streams[istream], cudaStreamNonBlocking);
    cudaMalloc((void**)&d_mats[istream], sizes);
  }

  // Start timer
  cudaDeviceSynchronize();
  start = std::chrono::high_resolution_clock::now();
  
  // USE STREAMS: Copy H2D, kernel, copy D2H
  //==============================================================
  for (int istream(0); istream < nstream; ++istream) {
    cudaMemcpyAsync(d_mats[istream], h_mat + istream * ncells,
                    sizes, cudaMemcpyHostToDevice, streams[istream]);
    dim3 blocks(TILEX,TILEY);
    dim3 grids((ncols + TILEX - 1)/TILEX, (nrows + TILEY - 1)/TILEY);
    multByTwo<<<grids, blocks, 0, streams[istream]>>>(nrows, ncols, d_mats[istream]);
    cudaMemcpyAsync(h_mat + istream * ncells, d_mats[istream],
                    sizes, cudaMemcpyDeviceToHost, streams[istream]);
  }

  // Sync host and device
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  std::cout << "With concurrency compute time (ms): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
            << std::endl;
  
  // Clean up
  for (int istream(0); istream < nstream; ++istream) {
    cudaFree(d_mats[istream]);
    cudaStreamDestroy(streams[istream]);
  }
  cudaFreeHost(h_mat);

  return 0;
}
