#include <iostream>
#include <math.h>
#include <vector>
//#include <opencv2/opencv.hpp>
#ifdef DP
typedef double REAL;
#else
typedef float REAL;
#endif

#define vec1di std::vector<int>
#define vec2di std::vector<std::vector<int>>
#define vec1df std::vector<float>
#define vec2df std::vector<std::vector<float>>

// Create sobel filter with size k x k
void sobel_filter(int k, REAL *&A) {
    //vec2df out(k);
    float v, x_dist, y_dist;
    for (int i = 0; i < k; i++) {
        //vec1df row(k);
        for (int j = 0; j < k; j++) {
            if (j == floor(k/2)){
                v = 0;
            }
            else {
                y_dist = (i - floor(k/2));
                x_dist = (j - floor(k/2));
                v = x_dist / (x_dist * x_dist + y_dist * y_dist);
            }
            //row[j] = v;
 		A[i * k + j] = v;
        }
        //out[i] = row;
    }
    //return out;
}

// Create padded input vector padded by k pixels
vec2di array_padding(int k, const vec2di &array) {
    size_t out_size = array.size() + 2*k;
    vec2di out(out_size);
    for (int r = 0; r < out_size; r++) {
        vec1di row(array.size() + 2*k, 0);
        if ((r > k) && (r < array.size() + k)) {
            for (int x = 0; x < array[0].size(); x++ ) {
                row[x+k] = array[r-k][x];
            }
        }
        out[r] = row;
    }
    return out;
}

void print_array(REAL *&A, int M) {
    /*for (int i = 0; i < X.size(); i++) {
        std::cout << "[";
       for (int j = 0; j < X[0].size(); j++) {
            std::cout << X[i][j] << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;*/
    std::cout << "[";
    for (int i = 0; i < M*M; i++) {

        std::cout << A[i] << " ";
        if ((i+1)%M ==0){
            std::cout <<"]\n";
            std::cout << "[";
        }

    }
}


// Do the product convolution of A by the Kernel
void fill(REAL *&A, int M) {
    REAL array [M*M] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 2, 5, 2, 0, 5, 2, 5, 2, 0, 5, 2, 5, 2, 0};
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
        A[i * M + j] = array[i * M + j];
        }
    }

}

void convolution_product(REAL *&array , REAL *&kernel, REAL *&out, int M, int K) {
  /*for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T tmp = T(0.0);
      for (int k = 0; k < K; k++) {
        tmp += A[i * N + k] * B[k * M + j];
      }
      C[i * M + j] = tmp;
    }
  }*/
  int total, elem = 0;
  for (int i = 1; i < M-1; i++) {
        for (int j = 1; j < M-1; j++) {
            total = 0;
            // Go through each element in the kernel array
            for (int x = 0; x < K; x++) {
                for (int y = 0; y < K; y++) {
                    elem = array[i * K +j];
                    std::cout << elem ;
                    total = total + elem * kernel[x * K + y]; // Add to the total value for the output pixel
                }
            }
            std::cout << "\n"<< total<< "\n" ;
            out[i * M + j] = total;
        }
    }
}

// TODO: Add arguments (image path, kernel size)
int main() {
    int k = 3;
    int M = 5;
    //int k = floor(kernel_size / 2);
    // TODO: Replace with opening an image with opencv
    //vec2di array = {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {5, 2, 5, 2, 0}, {5, 2, 5, 2, 0}, {5, 2, 5, 2, 0}};
    //vec2di out = array;
    //REAL *A = new REAL[M*K]; M = 5 K = 5 
    
    REAL *A = new REAL[M*M];
    //array = array_padding(k, array);
    fill(A,M);

    std::cout << "Input" << std::endl;
    print_array(A,M);
    
    std::cout << "]\n";
    REAL *kernel = new REAL[k*k];
    sobel_filter(k,kernel);
    
    std::cout << "Kernel" << std::endl;
    print_array(kernel, k);

    std::cout << "]\n";
    REAL *out = new REAL[M*M];
    convolution_product(A, kernel, out, M, k);

    std::cout << "Output" << std::endl;
    print_array(out,M);
   /*
    float total = 0;
    float elem = 0;
    // Go through each pixel in the original array
    for (int i = 0; i < out.size(); i++) {
        for (int j = 0; j < out[0].size(); j++) {
            total = 0;
            // Go through each element in the kernel array
            for (int x = 0; x < kernel.size(); x++) {
                for (int y = 0; y < kernel[0].size(); y++) {
                    elem = array[i + x][j + y];
                    total += elem * kernel[x][y]; // Add to the total value for the output pixel
                }
            }
            out[i][j] = total;
        }
    }
    // TODO: Replace with visualizing image with opencv
    std::cout << "Output array" << std::endl;
    print_array(out);
    */
    return 0;
}


