// -----------------------------------------------------------------------------
// * Name:       main_gpu.cxx
// * Purpose:    Driver for matrix multiplication on GPU
// * History:    Christophe Picard, Fall 2021
// -----------------------------------------------------------------------------

// includes, system
#include <cmath>
#include <iostream>
#include <string>

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


__global__ void
conv_naive( float* output, float* array, float* kernel, int w, int k)
{
  /* Naive function for calculating convolution between array and 
  * 2D kernel. In future requires tiling the image and the kernel 
  * into smaller pieces before calculating
  *
  * float* output:   output array
  * float* array:    padded input array
  * float* kernel:   the filter kernel
  * int w:           width of the non-padded input array
  * int k:           kernel width
  */ 

  // thread indexing
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int w_pad = w + k - 1;

  float accu = 0.0;
  float elem = 0.0;
  int test = 5;
  // Go through each element in the filter kernel
  for(int y=0; y<k; y++){
    for(int x=0; x<k; x++){
        // start from (row-k, col-k) position and move through the
        // elements in the kernel
        accu += array[(i + y) * w_pad + j + x] * kernel[x * k + y];

        // Debugging
        // if ((row == 0) && (col == 0))
        //   printf("%f * %f = %f\n", array[(row + y) * w_pad + col + x], kernel[x * k + y], array[(row + y) * w_pad + col + x] * kernel[x * k + y]);
      }
  }
  output[i * w + j] = accu;
}


__global__ void
conv_tiled( float* output, float* array, float* kernel, int w, int h, int k)
{
  /*
  Function that calculates the convolution between an array array and filter B.
  The convolution is done in tiles to save global memory access cost.
  Parameters:
    - float* output  Output filtered array
    - float* array   Padded input array
    - float* kernel  Used filter array
    - int w          Width of the non-padded input array
    - int h          Height of the non-padded input array
    - int k          Width of the kernel

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
  int pad = (k - 1) / 2; // Kernel has to always be odd numbered

  // row and col moving forward TW - 2*pad indices per tile
  //   by * (TW - 2*pad) removed from row to get current tile's row index
  //   bx * (TW - 2*pad) removed from col to get current tile's col index
  //     -> row index * width + col index results in correct indexing for the subtile
  size_t ind = (i - by * (TW - 2*pad)) * (w + 2*pad) + j - bx * (TW - 2*pad);
  subTile[ty][tx] = array[ind];

  // Debugging
  // printf("(%d, %d) (%d, %d) Subtile [%d,%d] = array[%d]\n", i, j, bx, by, ty, tx, ind);

  // Sync so all data in subtile is present for calculations. Later there is no need 
  // for another synchronisation because the tiles are independent of each other
  __syncthreads();

  for(int y=0; y<k; y++){
    for(int x=0; x<k; x++){
      // start from (row-k, col-k) position and move through the
      // elements in the kernel
      accu += subTile[ty + y][tx + x] * kernel[y * k + x];
      //accu += array[(row + y) * w_pad + col + x] * kernel[x * k + y];

      // Debugging
      // if ((ty == 0) && (tx == 1))
      //   printf("%d,%d: %f * %f = %f (%d)\n", bx, by, subTile[ty + y][tx + x], kernel[y * k + x], subTile[ty + y][tx + x] * kernel[y * k + x], y * k + x);
    }
  }
  // Only the center elements of convolution are needed, skip padded area around calculation
  if ((ty < TW - 2*pad) && (tx < TW - 2*pad)) {
    int outx = (by * (TW - 2*pad) + ty) * w;
    int outy = bx * (TW - 2*pad) + tx;
    // Do not write elements that are outside of output bounds because of tiling
    if ((outx < w) || (outy < h)) {
      // Index calculation:
      //   Each row and column of a tile increases the index in that direction by 2*pad
      //   So add multiple of that to each row + the thread row and to each column + the thread column 
      output[(by * (TW - 2*pad) + ty) * w + bx * (TW - 2*pad) + tx] = accu;

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


void print_array(REAL *&A, int M) {
  std::cout << "[";
  for (int i = 0; i < M*M; i++) {
    std::cout << A[i] << " ";
    if ((i+1)%M ==0){
      std::cout <<"]\n";
      std::cout << "[";
    }
  }
}


///
/// Top level driver
///
int main(int argc, char **argv) {

  std::cout << "[Matrix Multiply Using CUDA] - Starting..." << std::endl;
  int k = 3;
  REAL *kernel = new REAL[k*k];
  sobel_filter(k, kernel);
  print_array(kernel, k);
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
  int WB = 3;
  int HA = 5;
  int HB = 3;
  int WC = WA;
  int HC = HA;

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
  unsigned int size_A = (WA + 2) * (HA+2);
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A = (float *)malloc(mem_size_A);
  unsigned int size_B = WB * HB;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B = (float *)malloc(mem_size_B);
  
  // initialize host memory
  //fill_random<REAL>(h_A, WA, HA);
  //fill_random<REAL>(h_B, WB, HB);
  
  float AA [7*7] = {0,0,0,0,0,0,0,
                    0,1,1,1,1,1,0,
                    0,1,1,1,1,1,0,
                    0,1,1,1,1,1,0,
                    0,1,1,1,1,1,0,
                    0,1,1,1,1,1,0,
                    0,0,0,0,0,0,0};

  float BB [3*3] = {-0.5, 0.0, 0.5, 
                    -1.0, 0.0, 1.0,
                    -0.5, 0.0, 0.5};
  h_A = AA;
  h_B = BB;


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
  float *h_C = new float[25];//(float *)malloc(mem_size_C);

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

  // execute the tiled kernel
  threads = dim3(TW, TW);
  grid = dim3(3, 3); 
  conv_tiled<<<grid, threads >>>(d_C, d_A, d_B, 5, 5, 3);
  gpuErrchk(cudaPeekAtLastError());
  
  // copy result from device to host
  gpuErrchk(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

  
  // stop and destroy timer
  cudaEventCreate(&stop);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);

  print_array(h_A, 7);
  std::cout << std::endl;
  print_array(h_C, 5);

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
