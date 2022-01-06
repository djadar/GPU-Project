#include <iostream>
#include <math.h>
#include <vector>
#include <chrono>
#include <unistd.h>
//#include <opencv2/opencv.hpp>

using namespace std;
using namespace chrono;

#define REAL float

// Create sobel filter with size k x k
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

void array_padding(REAL *&A, REAL *&B, int pad, int w, int h) {
    int out_w = w + pad * 2;
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            B[(r + pad) * out_w + c + pad] = A[r * w + c];
        }
    }
}

void print_array(REAL *&A, int M) {
    std::cout << "[";
    for (int i = 0; i < M*M; i++) {
         
        std::cout << A[i] << " ";
        if ((i+1)%M == 0){
            std::cout << "]\n";
            std::cout <<  "[";
        }
    }
    std::cout << std::endl;
}

// TODO: Add arguments (image path, kernel size)
int main() {
    // Filter parameters
    int k = 3;
    int pad = floor(k / 2);

    // TODO: Replace with opening an image with opencv and using real widths and heights
    int w = 5;
    int h = 5; 
    REAL AA [5*5] = {1,1,1,1,1,
                     1,1,1,1,1,
                     1,1,1,1,1,
                     1,1,1,1,1,
                     1,1,1,1,1};
    REAL A [7*7] = { };
    REAL B[k*k];
    REAL C[5*5];

    // Array initializations
    float *a_nopad = new float [sizeof(float)*5*5];
    float *array = new float [sizeof(float)*7*7];
    float *kernel = new float [sizeof(float)*k*k];
    float *out = new float [sizeof(float)*5*5];

    a_nopad = AA;
    array = A;
    kernel = B;
    out = C;

    // Create padded array and filter array and print them for debugging
    array_padding(a_nopad, array, pad, 5, 5);
    sobel_filter(k, kernel);
    std::cout << "Kernel" << std::endl;
    print_array(kernel, k);
    std::cout << "Padded input array" << std::endl;
    print_array(array, 7);

    float total = 0;
    float elem = 0;
    int w_pad = w + 2 * pad;

    auto start = chrono::steady_clock::now();
    // Go through each pixel in the original array
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            total = 0;
            // Go through each element in the kernel array
            for (int x = 0; x < k; x++) {
                for (int y = 0; y < k; y++) {
                    elem = array[(r + y) * w_pad + c + x];
                    total += elem * kernel[y * k + x]; // Add to the total value for the output pixel
                }
            }
            out[r * w + c] = total;
        }
    }
    auto end = chrono::steady_clock::now();
    // TODO: Replace with visualizing image with opencv
    std::cout << "Filtered output array" << std::endl;
    print_array(out, 5);
    std::cout << "Elapsed time in microseconds: "
              << chrono::duration_cast<chrono::microseconds>(end - start).count()
              << " ms " << std::endl;

    return 0;
}