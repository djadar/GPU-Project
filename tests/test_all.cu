// -----------------------------------------------------------------------------
// * Name:       main_gpu.cxx
// * Purpose:    Driver for matrix multiplication on GPU
// * History:    Christophe Picard, Fall 2021
// -----------------------------------------------------------------------------

// includes, system
#include <cmath>
#include <ctime>

#include <iostream>
#include <string>

#include <random>
#include <limits>

#include <cuda.h>

// Parsing command line options using cxxopts 
// https://github.com/jarro2783/cxxopts.git
//#include "args.hxx"

// Matrix manipulation function
//#include "matrix_utils.h"

// Define different gemm kernel
// #include <gemm_kernel.cuh>
// #include <conv_naive.cuh>
// #include <conv_tiled.cuh>

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

#define REAL float
#define BLOCK_SIZE 1
#define TW 4
#define WIDTH_K 3


/// ----------------------------------------------------------------------------
/// \fn void init_mat( int N, T *&A, T *&B)
/// \brief Set matrix coefficients
/// \param A First matrix to initialize 
/// \param B Second matrix to initialize 
/// \param N Size of the matrix
/// ----------------------------------------------------------------------------
template <typename T> 
void fill_random(T *&A, int N, int M) {
  std::mt19937 e(static_cast<unsigned int>(std::time(nullptr)));
  std::uniform_real_distribution<T> f;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      A[i * M + j] = f(e);
    }
  }
}


void conv(REAL *&out, REAL *&A, REAL *&K, int wK, int wA, int hA){
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
    //printf("(%d, %d): %f \n", row, col, accu);
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

  // kernel to shared memory
  // Faster: some threads load 2 all threads calculate

  // Shared tile
  __shared__ float subTile[TW][TW];
  // TODO: add kernel to shared memory ----------
  
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

  // Debugging
  // printf("(%d, %d) (%d, %d) Subtile [%d,%d] = A[%d]\n", i, j, bx, by, ty, tx, ind);

  // Sync so all data in subtile is present for calculations. Later there is no need 
  // for another synchronisation because the tiles are independent of each other
  __syncthreads();

  for(int y=0; y<wK; y++){
    for(int x=0; x<wK; x++){
      // start from (row-k, col-k) position and move through the
      // elements in the kernel
      accu += subTile[ty + y][tx + x] * K[y * wK + x];

      // Debugging
      // if ((ty == 0) && (tx == 1))
      //   printf("%d,%d: %f * %f = %f (%d)\n", bx, by, subTile[ty + y][tx + x], K[y * wK + x], subTile[ty + y][tx + x] * K[y * wK + x], y * wK + x);
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

      // Debugging
      // printf("%d, %d :: %d, %d\n", ty, tx, (by * (TW - 2*pad) + ty) * w, bx * (TW - 2*pad) + tx);
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
    - int wK        Width of the kernel

  */

  // kernel to shared memory
  // Faster: some threads load 2 all threads calculate

  // Shared tile
  __shared__ float subTile[TW][TW];
  __shared__ float kernel[WIDTH_K][WIDTH_K];
  // TODO: add kernel to shared memory ----------
  
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

  // Debugging
  // printf("(%d, %d) (%d, %d) Subtile [%d,%d] = A[%d]\n", i, j, bx, by, ty, tx, ind);

  // Sync so all data in subtile is present for calculations. Later there is no need 
  // for another synchronisation because the tiles are independent of each other
  __syncthreads();

  for(int y=0; y<WIDTH_K; y++){
    for(int x=0; x<WIDTH_K; x++){
      // start from (row-k, col-k) position and move through the
      // elements in the kernel
      accu += subTile[ty + y][tx + x] * kernel[y][x];

      // Debugging
      // if ((ty == 0) && (tx == 1))
      //   printf("%d,%d: %f * %f = %f (%d)\n", bx, by, subTile[ty + y][tx + x], K[y * WIDTH_K + x], subTile[ty + y][tx + x] * K[y * WIDTH_K + x], y * wK + x);
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

      // Debugging
      // printf("%d, %d :: %d, %d\n", ty, tx, (by * (TW - 2*pad) + ty) * w, bx * (TW - 2*pad) + tx);
    }
  }
}


void sobel_filter(int k, REAL *&A) {
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


///
/// Top level driver
///
int main(int argc, char **argv) {

  std::cout << "[Matrix Multiply Using CUDA] - Starting..." << std::endl;
  int k = 3;
  REAL *kernel = new REAL[k*k];
  sobel_filter(k, kernel);
  print_array(kernel, k, k);
  // Define parser 
  // args::ArgumentParser parser("gemm_cuda", "Matrix Multiply using CUDA");

  // // Set parser value
  // args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  // args::ValueFlag<int> widthA(parser, "widthA", "Width of matrix A", {"wA"}, 5);
  // args::ValueFlag<int> widthB(parser, "widthB", "Width of matrix B", {"wB"}, 3);
  // args::ValueFlag<int> heightA(parser, "heightA", "Height of matrix A", {"hA"}, 5);
  // args::ValueFlag<int> heightB(parser, "heightB", "Height of matrix B", {"hB"}, 3);
  // //args::ValueFlag<int> blockSize(parser, "blockSize", "Size of blocks", {"sb"}, 32);

  // // Invoke parser
  // try {
  //   parser.ParseCLI(argc, argv);
  // } catch (args::Help) {
  //   std::cout << parser;
  //   return 0;
  // } catch (args::ParseError e) {
  //   std::cerr << e.what() << std::endl;
  //   std::cerr << parser;
  //   return 1;
  // } catch (args::ValidationError e) {
  //   std::cerr << e.what() << std::endl;
  //   std::cerr << parser;
  //   return 1;
  // }

  // Initialize matrix dimensions
  // int WA = args::get(widthA);
  // int WB = args::get(widthB);
  // int HA = args::get(heightA);
  // int HB = args::get(heightB);
  //int BS = args::get(blockSize);



  int WA = 5;
  int WK = 3;
  int HA = 5;
  int HK = 3;
  int WC = WA;
  int HC = HA;
  WA = WA + WK - 1;
  HA = HA + WK - 1;

  // Setup CUDA environnement 
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

  // utilities
  cudaEvent_t start;
  cudaEvent_t stop;
  float msecTotal;

  // allocate host memory for matrices A and B
  unsigned int size_A = (WA) * (HA);
  unsigned int mem_size_A = sizeof(float) * size_A;
  // float *h_A = (float *)malloc(mem_size_A);
  unsigned int size_B = WK * HK;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B = (float *)malloc(mem_size_B);
  
  // initialize host memory
  //fill_random<REAL>(h_A, WA, HA);
  //fill_random<REAL>(h_B, WB, HB);
  
  // float AA [7*8] = {0,0,0,0,0,0,0,0,
  //                   0,1,1,1,1,1,1,0,
  //                   0,1,1,1,1,1,1,0,
  //                   0,1,1,1,1,1,1,0,
  //                   0,1,1,1,1,1,1,0,
  //                   0,1,1,1,1,1,1,0,
  //                   0,0,0,0,0,0,0,0};

  float BB [3*3] = {-0.5, 0.0, 0.5, 
                    -1.0, 0.0, 1.0,
                    -0.5, 0.0, 0.5};
  // h_A = AA;
  h_B = BB;

  float *h_A = new REAL[WA*HA];
  fill_random<REAL>(h_A, WA, HA);


  // allocate device memory
  float *d_A;
  gpuErrchk(cudaMalloc((void **)&d_A, mem_size_A));
  float *d_B;
  gpuErrchk(cudaMalloc((void **)&d_B, mem_size_B));


  // allocate device memory for result
  unsigned int size_C = WC * HC;

  unsigned int mem_size_C = sizeof(float) * size_C;
  float *d_C;
  gpuErrchk(cudaMalloc((void **)&d_C, mem_size_C));

  // allocate host memory for the result
  float *h_C = new float[WC*HC];//(float *)malloc(mem_size_C);

  dim3 threads, grid;

  // create and start timer
  cudaEventCreate(&start);
  cudaEventRecord(start, NULL);
 
  // copy host memory to device
  gpuErrchk(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

  // setup execution parameters
  // threads = dim3(BS, BS);
  // threads=dim3(BLOCK_SIZE, BLOCK_SIZE);
  // grid = dim3(WC / threads.x, HC / threads.y);

  // TODO: Find programmatic values for grids/threads. Currently is calculated only for this example
  // execute the naive kernel
  // threads = dim3(5, 5);
  // grid = dim3(1); 
  // conv_naive<<<grid, threads>>>(d_C, d_A, d_B, 5, 3);




  print_array(h_A, WA, HA);
  conv(h_C, h_A, h_B, WK, WC, HC);
  print_array(h_C, WC, HC);

  
  threads = dim3(TW, TW);
  grid = dim3(3, 3);

  // conv_naive<<<grid, threads >>>(d_C, d_A, d_B, WK, WC, HC);
  // gpuErrchk(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
  // print_array(h_C, WC, HC);

  conv_tiled<<<grid, threads >>>(d_C, d_A, d_B, WK, WC, HC);
  gpuErrchk(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
  print_array(h_C, WC, HC);
  std::cout << "ok";

  // conv_tiled_shared<<<grid, threads >>>(d_C, d_A, d_B, WC, HC);
  // gpuErrchk(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
  // print_array(h_C, WC, HC);

  gpuErrchk(cudaPeekAtLastError());
  
  // stop and destroy timer
  cudaEventCreate(&stop);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  std::cout << "ok2";

  std::cout << std::endl;

	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );

  // A and B are static for now
  free(h_C);

  /* Performance computation, results and performance printing ------------ */
  auto flop = 2 * (float)WC * (float)HC * (float)WA;

  std::cout << " == Performances " << std::endl;
  std::cout << "\t Processing time: " << msecTotal << " (ms)"
            << std::endl;
  std::cout << "\t GFLOPS: " << flop / msecTotal / 1e+6 << std::endl;

  return (EXIT_SUCCESS);
}
