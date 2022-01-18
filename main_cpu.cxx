
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
#ifdef DP
typedef double REAL;
#else
typedef float REAL;
#endif
#define check_out 1

/*----------------------------------------------------------------------------*/
/* Toplevel function.                                                         */
/*----------------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
  std::cout << "[Edge detection Using CPU]" << std::endl;

  // Define parser 
  args::ArgumentParser parser("gemm_cpu", "Matrix Multiply using CPU");

  // Set parser value
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag<int> widthC(parser, "widthC", "Width of output matrix C", {"WC"},
                              256);
  args::ValueFlag<int> heightA(parser, "heightC", "Height of output matrix C", {"HC"},
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
  int WA, WB, HA, HB, WC, HC;
  WC = args::get(widthC);
  HC = args::get(heightC);
  WK = args::get(widthK);
  WA = WC + WK -1 ;
  HA = HC + WK -1;

  // Initialisation of matrix input and the kernel
  REAL *h_K = new REAL[WK*WK];
  sobel_filter(WK, h_K);

  // allocate host memory for the result
  float *h_C = new float[WC*HC];

  // --- Begin calculations ---
 
  // Run CPU convolution
  std::cout << " CPU convolution of matrice output"
            << " of size " << WC << "x" << HC << " with a Sobel Filter of size " << WK << "x" << WK
            << std::endl;

  std::cout << " == Computation starts..." << std::endl;

  auto start = std::chrono::system_clock::now();

  // Print kernel and input
  print_array(h_K, WK, WK);
  print_array(h_A, WA, HA);
  
  conv_cpu(h_C, h_A, h_K, WK, WC, HC);
  
  auto elapse = std::chrono::system_clock::now() - start;
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapse);

  // Print output
  print_array(h_C, WC, HC);

  /* Performance computation, results and performance printing ------------ */
  auto flop = 2 * M * N * K ;

  std::cout << " == Performances " << std::endl;
  std::cout << "\t Processing time: " << duration.count() << " (ms)"
            << std::endl;
  std::cout << "\t GFLOPS: " << flop / duration.count() / 1e+6 << std::endl;

  /*if (check_out)
    check_result<REAL>(A, B, C, M, N, K); // Res checking
*/
  free(h_C);
  /* End of the sequential program ------------------------------------------ */
  return (EXIT_SUCCESS);
}