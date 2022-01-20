// -----------------------------------------------------------------------------
// * Name:       main_gpu.cxx
// * Purpose:    Driver for matrix convolutional product on GPU
// -----------------------------------------------------------------------------

// includes, system
#include <cmath>
#include <iostream>
#include <string>

#include <cuda.h>

#include "args.hxx"

// Matrix manipulation function
#include "utils.h"

// Define different convolution kernel
#include <conv_naive.cuh>
#include <conv_shared.cuh>


// Constants
#define REAL float

#define WIDTH_K 3

///
/// Top level driver
///
int main(int argc, char **argv) {

  std::cout << "[Matrix Convolutional Product Using CUDA] - Starting..." << std::endl;

  // Define parser 
  args::ArgumentParser parser("edge_cuda", "Matrix Convolutional Product Using CUDA");

  // Set parser value
  // Set parser value
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag<int> widthC(parser, "widthC", "Width of output matrix C", {"WC"},
                              256);
  args::ValueFlag<int> heightC(parser, "heightC", "Height of output matrix C", {"HC"},
                               256);
  args::ValueFlag<int> widthK(parser, "widthB", "Width of kernel matrix K", {"WK"},
                              3);
  args::ValueFlag<int> choice(parser, "choice", "Choose the way of doing the calculation 1 : conv_naive ; 2: conv_tiled ; 3: conv_shared", {"choice"},
                              1);
  // Invoke parser
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  } catch (args::ValidationError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;}
  /*}catch (args::get(choice) !=1 ||args::get(choice) !=2 || args::get(choice) !=3) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;}*/

  int n = args::get(choice);
  // Initialize matrix dimensions
  int WA, WK, HA, WC, HC;
  WC = args::get(widthC);
  HC = args::get(heightC);
  WK = args::get(widthK);
  WA = WC + WK -1 ;
  HA = HC + WK -1;

  // Setup CUDA environnement 
  cudaError_t error;

  cudaDeviceProp deviceProp;
  int devID = 0;
  error = cudaGetDevice(&devID);

  if (error != cudaSuccess) {
    printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
  }

  error = cudaGetDeviceProperties(&deviceProp, devID);

  if (deviceProp.computeMode == cudaComputeModeProhibited) {
    std::cerr << "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice() ." <<std::endl;
    exit(EXIT_SUCCESS);
  }

  if (error != cudaSuccess) {
    printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
  } else {
    printf("\t GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
  }

  // utilities
  cudaEvent_t start;
  cudaEvent_t stop;
  float msecTotal;

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
  cudaMalloc((void **)&d_A, mem_size_A);
  float *d_K;
  cudaMalloc((void **)&d_K, mem_size_K);
  float *d_C;
  cudaMalloc((void **)&d_C, mem_size_C);
  

  // --- Begin calculations ---


  // setup execution parameters
  dim3 threads, grid;
  threads = dim3(TW, TW);
  
  // Print kernel and input
  print_array(h_K, WK, WK);
  print_array(h_A, WA, HA);

  // create and start timer
  cudaEventCreate(&start);
  cudaEventRecord(start, NULL);

  // copy host memory to device
  cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K, mem_size_K, cudaMemcpyHostToDevice);

  // Run GPU convolutions
  if (n==1){
      int blocksX = WC / (TW) + 1;
      int blocksY = HC / (TW) + 1;
      grid = dim3(blocksX, blocksY);
      conv_naive<<<grid, threads >>>(d_C, d_A, d_K, WK, WC, HC);
      std::cout << " ================ NAIVE ===================" << std::endl;
    }else if (n==2){
      int blocksX = WC / (TW - WK + 1) + 1;
      int blocksY = HC / (TW - WK + 1) + 1;
      grid = dim3(blocksX, blocksY);
      conv_shared<<<grid, threads >>>(d_C, d_A, d_K, WC, HC);
      std::cout << " ================ SHARED ===================" << std::endl;
    }
  cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

  // stop and destroy timer
  cudaEventCreate(&stop);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);

  

  //Print output
  print_array(h_C, WC, HC);  
  
  // Performance computation, results and performance printing
  cudaEventElapsedTime(&msecTotal, start, stop);
  auto flop = 2 * (float)WC * (float)HC * (float)WK * (float)WK;
  int microsecond = msecTotal * 1000;
  std::cout << " == Performances " << std::endl;
  std::cout << "\t Processing time: " << microsecond << " (µs)"
            << std::endl;
  std::cout << "\t GFLOPS: " << flop / microsecond / 1e+3 << std::endl;

	cudaFree( d_A );
	cudaFree( d_K );
	cudaFree( d_C );
  free(h_A);
  free(h_K);
  free(h_C);

  return (EXIT_SUCCESS);
}
