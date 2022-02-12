
// -----------------------------------------------------------------------------
// * Name:       main_cpu.cxx
// * Purpose:    Driver for matrix convolutional product on CPU
// -----------------------------------------------------------------------------

#include <chrono>
#include <cmath>
#include <iostream>

// #include <typeinfo>

#include "args.hxx"

//#include "gemm_noblas.h"
#include "utils.h"

/*----------------------------------------------------------------------------*/
/* Floating point datatype and op                                             */
/*----------------------------------------------------------------------------*/
#define REAL float
#define check_out 1

/*----------------------------------------------------------------------------*/
/* Toplevel function.                                                         */
/*----------------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
  //std::cout << "[Matrix Convolutional Product Using CPU]" << std::endl;

  // Define parser 
  args::ArgumentParser parser("edge_cpu", "Matrix Convolutional Product Using CPU");

  // Set parser value
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag<int> widthC(parser, "widthC", "Width of output matrix C", {"WC"},
                              256);
  args::ValueFlag<int> heightC(parser, "heightC", "Height of output matrix C", {"HC"},
                               256);
  args::ValueFlag<int> widthK(parser, "widthB", "Width of kernel matrix K", {"WK"},
                              3);

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
    return 1;
  }

  // Initialize matrix dimensions
  int WA, HA, WC, HC, WK;
  WC = args::get(widthC);
  HC = args::get(heightC);
  WK = args::get(widthK);
  WA = WC + WK -1 ;
  HA = HC + WK -1;

  // Initialisation of matrix input and the kernel
  float *h_A = new REAL[WA*HA];
  fill_random<REAL>(h_A, WA, HA);

  REAL *h_K = new REAL[WK*WK];
  sobel_filter(WK, h_K);

  // allocate host memory for the result
  float *h_C = new float[WC*HC];

  // --- Begin calculations ---
 
  // Run CPU convolution
  //std::cout << " CPU convolution of matrice output"
            //<< " of size " << WC << "x" << HC << " with a Sobel Filter of size " << WK << "x" << WK
            //<< std::endl;

  //std::cout << " == Computation starts..." << std::endl;

  // Print kernel and input
  //print_array(h_K, WK, WK);
  //print_array(h_A, WA, HA);
  
  auto start = std::chrono::system_clock::now();
  
  conv_cpu(h_C, h_A, h_K, WK, WC, HC);
  
  auto elapse = std::chrono::system_clock::now() - start;
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(elapse);

  // Print output
  //print_array(h_C, WC, HC);

  /* Performance computation, results and performance printing ------------ */
  auto flop = 2 * WC * HC * WK * WK ;

 // std::cout << " == Performances " << std::endl;
  std::cout << "Processing time: " << duration.count() << " (µs)"
            << std::endl;
  std::cout << "flop: " << flop << std::endl;
  std::cout << "ok: " << flop / duration.count() << std::endl;
  std::cout << "GFLOPS: " << flop / duration.count() / 1e+3 << std::endl;

  /*if (check_out)
    check_result<REAL>(A, B, C, M, N, K); // Res checking
*/
  free(h_A);
  free(h_K);
  free(h_C);
  /* End of the sequential program ------------------------------------------ */
  return (EXIT_SUCCESS);
}