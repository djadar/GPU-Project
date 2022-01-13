// includes, system
#include <cmath>
#include <ctime>

#include <iostream>
#include <string>

#include <random>
#include <limits>

#include <cuda.h>


// Constants
#define REAL float
#define BLOCK_SIZE 1
#define TW 4
#define WIDTH_K 3


// Define error checking 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/// ----------------------------------------------------------------------------
/// \fn gpuAssert(cudaError_t code, const char *str, int line, bool abort=true)
/// @param[in] code (cudaError_t): error code from Cuda
/// @param[in] file (const char *): name of the file containing the error
/// @param[in] line (int): line number containing the error
/// @param[in] abort (bool): force abort on cuda error
/// ----------------------------------------------------------------------------
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    std::cerr << "CUDA error: "<< cudaGetErrorString(code)<<" "<< file<<" "<< line<<std::endl;
    if (abort) exit(code);
  }
}


template <typename T> 
void fill_random(T *&A, int N, int M) {
  /* Randomly fill array A with size N * M 
  */
  std::mt19937 e(static_cast<unsigned int>(std::time(nullptr)));
  std::uniform_real_distribution<T> f;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      A[i * M + j] = f(e);
    }
  }
}


void conv_cpu(REAL *&out, REAL *&A, REAL *&K, int wK, int wA, int hA){
    /* Calculate convolution on array A with a filter K sequentially on the CPU
    * Parameters:
    * REAL *&out    Output filtered array
    * REAL *&A      Input array to be filtered
    * REAL *&K      Used filter kernel for convolution
    * int wK        Width of the filter kernel
    * int wA        Width of the non-padded input array
    * int hA        Height of the non-padded input array
    */
    float total = 0;
    float elem = 0;
    int w_pad = wA + wK - 1;
    // Go through each pixel in the original array
    for (int r = 0; r < hA; r++) {
        for (int c = 0; c < wA; c++) {
            total = 0;
            // Go through each element in the kernel array
            for (int x = 0; x < wK; x++) {
                for (int y = 0; y < wK; y++) {
                    elem = A[(r + y) * w_pad + c + x];
                    total += elem * K[y * wK + x]; // Add to the total value for the output pixel
                }
            }
            out[r * wA + c] = total;
        }
    }
}

__global__ void
conv_naive( float* out, float* A, float* K, int wK, int wA, int hA)
{
  /* Naive function for calculating convolution between array and 
  * 2D kernel. In future requires tiling the image and the kernel 
  * into smaller pieces before calculating
  *
  * float* out:   output array
  * float* A:     padded input array A
  * float* K:     the filter kernel array
  * int wK:       width of the filter kernel
  * int wA:       width of the input array
  * int hA:       height of the input array
  */

  int pad = wK / 2;
  int w_pad = wA + pad * 2;
  // thread indexing
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float accu = 0.0;
  // Go through each element in the filter kernel
  for(int y=0; y<wK; y++){
    for(int x=0; x<wK; x++){
        // start from (row-k, col-k) position and move through the
        // elements in the kernel
        accu = accu + A[(row + y)*w_pad + col + x] * K[y * wK + x];
      }
  }
  // each thread writes one element to output matrix
  if ((row < hA) && (col < wA)) {
    out[ row * wA + col ] = accu;
  }
}


__global__ void
conv_tiled( float* out, float* A, float* K, int wK, int wA, int hA)
{
  /*
  Function that calculates the convolution between an array array and filter B.
  The convolution is done in tiles to save global memory access cost.
  Parameters:
    - float* out    Output filtered array
    - float* A      Padded input array
    - float* K      Used filter array
    - int wA        Width of the non-padded input array
    - int hA        Height of the non-padded input array
    - int wK        Width of the kernel

  */

  // Shared tile array
  __shared__ float subTile[TW][TW];
  
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int i = by * TW + ty;
  int j = bx * TW + tx;

  float accu = 0.0;
  int pad = wK / 2; // Kernel has to always be odd numbered

  // row and col moving forward TW - 2*pad indices per tile
  //   by * 2*pad removed from row to get current tile's row index
  //   bx * 2*pad removed from col to get current tile's col index
  //     -> row index * width + col index results in correct indexing for the subtile
  size_t ind = (i - by * 2*pad) * (wA + 2*pad) + j - bx * 2*pad;
  subTile[ty][tx] = A[ind];

  // Sync so all data in subtile is present for calculations. Later there is no need 
  // for another synchronisation because the tiles are independent of each other
  __syncthreads();

  for(int y=0; y<wK; y++){
    for(int x=0; x<wK; x++){
      // start from (row-k, col-k) position and move through the
      // elements in the kernel
      accu += subTile[ty + y][tx + x] * K[y * wK + x];
    }
  }
  // Only the center elements of convolution are needed, skip padded area around calculation
  if ((ty < TW - 2*pad) && (tx < TW - 2*pad)) {
    int outy = (by * (TW - 2*pad) + ty);
    int outx = bx * (TW - 2*pad) + tx;
    // Do not write elements that are outside of output bounds because of tiling
    if ((outx < wA) && (outy < hA)) {
      // Index calculation:
      //   Each row and column of a tile increases the index in that direction by 2*pad
      //   So add multiple of that to each row + the thread row and to each column + the thread column 
      out[outy * wA + outx] = accu;
    }
  }
}


__global__ void
conv_tiled_shared( float* out, float* A, float* K, int wA, int hA)
{
  /*
  Function that calculates the convolution between an array array and filter B.
  The convolution is done in tiles to save global memory access cost.
  Parameters:
    - float* out    Output filtered array
    - float* A      Padded input array
    - float* K      Used filter array
    - int wA        Width of the non-padded input array
    - int hA        Height of the non-padded input array
  */
  // Shared tile and filter kernel array
  __shared__ float subTile[TW][TW];
  __shared__ float kernel[WIDTH_K][WIDTH_K];
  
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int i = by * TW + ty;
  int j = bx * TW + tx;

  float accu = 0.0;
  int pad = WIDTH_K / 2; // Kernel has to always be odd numbered

  // row and col moving forward TW - 2*pad indices per tile
  //   by * 2*pad removed from row to get current tile's row index
  //   bx * 2*pad removed from col to get current tile's col index
  //     -> row index * width + col index results in correct indexing for the subtile
  size_t ind = (i - by * 2*pad) * (wA + 2*pad) + j - bx * 2*pad;
  subTile[ty][tx] = A[ind];
  if ((ty < WIDTH_K) && (tx < WIDTH_K))
    kernel[ty][tx] = K[ty * WIDTH_K + tx];

  // Sync so all data in subtile is present for calculations. Later there is no need 
  // for another synchronisation because the tiles are independent of each other
  __syncthreads();

  for(int y=0; y<WIDTH_K; y++){
    for(int x=0; x<WIDTH_K; x++){
      // start from (row-k, col-k) position and move through the
      // elements in the kernel
      accu += subTile[ty + y][tx + x] * kernel[y][x];
    }
  }
  // Only the center elements of convolution are needed, skip padded area around calculation
  if ((ty < TW - 2*pad) && (tx < TW - 2*pad)) {
    int outy = (by * (TW - 2*pad) + ty);
    int outx = bx * (TW - 2*pad) + tx;
    // Do not write elements that are outside of output bounds because of tiling
    if ((outx < wA) && (outy < hA)) {
      // Index calculation:
      //   Each row and column of a tile increases the index in that direction by 2*pad
      //   So add multiple of that to each row + the thread row and to each column + the thread column 
      out[outy * wA + outx] = accu;
    }
  }
}


void sobel_filter(int k, REAL *&A) {
  /* 
  Function for creating a variable size sobel filter
  Parameters:
  - int k         Size of the sobel filter (k*k)
  - REAL *&A      Output sobel filter
  */
  float v, x_dist, y_dist;
  for (int i = 0; i < k; i++) {
      for (int j = 0; j < k; j++) {
          if (j == floor(k/2)){
              v = 0;
          }
          else {
              y_dist = (i - floor(k/2));
              x_dist = (j - floor(k/2));
              v = x_dist / (x_dist * x_dist + y_dist * y_dist);
          }
          A[i * k + j] = v;
      }
  }
}


void print_array(REAL *&A, int w, int h) {
  /* 
  Function for printing a 1D array to console
  Parameters:
  - REAL *&A      input array to be printed
  - int w         input array width
  - int h         input array height
  */
  std::cout << "[";
  for (int i = 0; i < w*h; i++) {
    if (i < w*h - 5)
      continue;
    std::cout << A[i] << " ";
    if ((i+1)%w ==0){
      std::cout <<"]\n";
      std::cout << "[";
    }
  }
  std::cout <<"\n";
}


int main(int argc, char **argv) {

  std::cout << "[Matrix Multiply Using CUDA] - Starting..." << std::endl;

  // Setup CUDA environment 
  cudaError_t error;

  cudaDeviceProp deviceProp;
  int devID = 0;
  error = cudaGetDevice(&devID);

  if (error != cudaSuccess) {
    printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
  }
  
  // info
  int dev_count;
  cudaGetDeviceCount( &dev_count);
  cudaDeviceProp dev_prop;
  for (int i = 0; i < dev_count; i++)
    cudaGetDeviceProperties( &dev_prop, i);

  error = cudaGetDeviceProperties(&deviceProp, devID);

  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    std::cerr << "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice() ." <<std::endl;
    exit(EXIT_SUCCESS);
  }

  if (error != cudaSuccess) {
    printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
  } else {
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
  }
  // -- END Cuda environment

  // TODO: Add as parameters
  int WC = 128;
  int HC = 256;
  int WK = 3;
  int WA = WC + WK - 1;
  int HA = HC + WK - 1;
  

  // allocate host memory for matrices A and K
  unsigned int mem_size_A = sizeof(float) * WA * HA;
  unsigned int mem_size_K = sizeof(float) * WK * WK;
  unsigned int mem_size_C = sizeof(float) * WC * HC;

  float *h_A = new REAL[WA*HA];
  fill_random<REAL>(h_A, WA, HA);

  REAL *h_K = new REAL[WK*WK];
  sobel_filter(WK, h_K);
  
  // allocate host memory for the result
  float *h_C = new float[WC*HC];

  // allocate device memory
  float *d_A;
  gpuErrchk(cudaMalloc((void **)&d_A, mem_size_A));
  float *d_K;
  gpuErrchk(cudaMalloc((void **)&d_K, mem_size_K));
  float *d_C;
  gpuErrchk(cudaMalloc((void **)&d_C, mem_size_C));
  // copy host memory to device
  gpuErrchk(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_K, h_K, mem_size_K, cudaMemcpyHostToDevice));


  // --- Begin calculations ---

  // Print kernel and input
  print_array(h_K, WK, WK);
  print_array(h_A, WA, HA);

  // Run CPU convolution
  conv_cpu(h_C, h_A, h_K, WK, WC, HC);
  std::cout << " ================ CPU ===================" << std::endl;
  print_array(h_C, WC, HC);

  // Run GPU convolutions
  dim3 threads, grid;
  threads = dim3(TW, TW);
  int blocksX = WC / (TW - 2) + 1;
  int blocksY = HC / (TW - 2) + 1;
  grid = dim3(blocksX, blocksY);

  conv_naive<<<grid, threads >>>(d_C, d_A, d_K, WK, WC, HC);
  gpuErrchk(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
  std::cout << " ================ NAIVE ===================" << std::endl;
  print_array(h_C, WC, HC);

  conv_tiled<<<grid, threads >>>(d_C, d_A, d_K, WK, WC, HC);
  gpuErrchk(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
  std::cout << " ================ TILED ===================" << std::endl;
  print_array(h_C, WC, HC);

  conv_tiled_shared<<<grid, threads >>>(d_C, d_A, d_K, WC, HC);
  gpuErrchk(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
  std::cout << " ================ SHARED ===================" << std::endl;
  print_array(h_C, WC, HC);

  gpuErrchk(cudaPeekAtLastError());

	cudaFree( d_A );
	cudaFree( d_K );
	cudaFree( d_C );
  free(h_A);
  free(h_K);
  free(h_C);

  return (EXIT_SUCCESS);
}
