// -----------------------------------------------------------------------------
// * Name:       utils.h
// * Purpose:    Provide a set of function to manipulate matrices 
// -----------------------------------------------------------------------------

#pragma once

/// I/O Libraries
#include <iostream>
#include <fstream>
#include <string>

/// Number manipulation
#include <limits>
#include <cmath>

/// Random number generation
#include <random>
#include <ctime>

#define REAL float
/// Only for checking
//#include "gemm_noblas.h"

/// ----------------------------------------------------------------------------
/// \fn void init_mat( int N, T *&A, T *&B)
/// \brief Set matrix coefficients
/// \param A First matrix to initialize 
/// \param B Second matrix to initialize 
/// \param N Size of the matrix
/// ----------------------------------------------------------------------------
template <typename T> void fill_random(T *&A, int N, int M) {
  std::mt19937 e(static_cast<unsigned int>(std::time(nullptr)));
  std::uniform_real_distribution<T> f;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      A[i * M + j] = f(e);
    }
  }
}

/// ----------------------------------------------------------------------------
/// \fn void sobel_filter(int k, REAL *&A)
/// \brief Function for creating a variable size sobel filter
/// \param A Output sobel filter 
/// \param k Size of the sobel filter (k*k)
/// ----------------------------------------------------------------------------

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

/// ----------------------------------------------------------------------------
/// \fn void print_array(REAL *&A, int w, int h)
/// \brief Print a 1D matrix in the terminal
/// \param A Matrix 
/// \param w Width of A
/// \param h height of A
/// ----------------------------------------------------------------------------
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



