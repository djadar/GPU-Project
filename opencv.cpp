
// -----------------------------------------------------------------------------
// * Name:       opencv.cpp
// * Purpose:    Testing edge detection filtering with opencv images
// -----------------------------------------------------------------------------

#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

#ifdef DP
typedef double REAL;
#else
typedef float REAL;
#endif

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

void print_array(char * name, REAL *&A, int M) {
    for (int i = 0; i < M*M; i++) { 
        std::cout << A[i] << " ";
        if ((i+1)%M ==0){
            std::cout <<"]\n";
            std::cout << "[";
        }
    }
    std::cout << "" << std::endl;
}

void array_padding(REAL *&A, REAL *&B, int pad, int w, int h) {
    int out_w = w + pad * 2;
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            B[(r + pad) * out_w + c + pad] = A[r * w + c];
        }
    }
}

void conv(REAL *&frameArray, REAL *&array, int * size, REAL *&kernel, int k, REAL *&out){
    array_padding(frameArray, array, k/2, size[0], size[1]);
    float total = 0;
    float elem = 0;
    int w_pad = size[1] + k - 1;
    // Go through each pixel in the original array
    for (int r = 0; r < size[0]; r++) {
        for (int c = 0; c < size[1]; c++) {
            total = 0;
            // Go through each element in the kernel array
            for (int x = 0; x < k; x++) {
                for (int y = 0; y < k; y++) {
                    elem = array[(r + y) * w_pad + c + x];
                    total += elem * kernel[y * k + x]; // Add to the total value for the output pixel
                }
            }
            out[r * size[1] + c] = total;
        }
    }
}

void save_result(char* name, REAL *&frameArray, int *size ){
    cv::Mat image ;             // input image
    image = cv::imread(name, cv::IMREAD_GRAYSCALE);

    int numCols = size[1];
    int numRows = size[0];
    REAL intensity;
    int value;
    for (int x = 0; x < numCols; x++) {          // x-axis, cols
        for (int y = 0; y < numRows; y++) { 
            intensity = frameArray[x * numCols + y];
           
            if (intensity >= 0 && intensity <= 255){
                value = intensity;
            }else if (intensity < 0){
                value = 0;
            }else if (intensity > 255){
                value = 255;
            }
            image.at<uchar>(x, y) = frameArray[x * numCols + y];
        }
    }
    std::cout <<" \n" << std::endl;

    cv::imwrite("result_image.jpg", image);
}

int main( int argc, char** argv ) {

    char* name = "smiley.jpg";

    int *size = (int *)malloc(sizeof(int)*2);
    cv::Mat image;
    image = cv::imread(name, cv::IMREAD_GRAYSCALE);

    if(! image.data ) {
        std::cout <<  "Image not found or unable to open" << std::endl ;
        return -1;
    }
    int numCols = image.cols;
    int numRows = image.rows;

    REAL *frameArray = new REAL[numCols*numRows];
    int intensity;
    for (int x = 0; x < numCols; x++) {          // x-axis, cols
        for (int y = 0; y < numRows; y++) {          // y-axis rows
            intensity = image.at<uchar>(cv::Point(y, x));
            frameArray[x * numRows + y] = intensity;
        }
    }
    size[0] = numCols;
    size[1] = numRows;

    //kernel
    int k = 3;
    int pad = floor(k / 2);
    REAL *kernel = new REAL[k*k];
    sobel_filter(k, kernel);

    //Array
    REAL *array = new REAL[(size[0]+1)*(size[1]+1)];
    REAL *out = new REAL[size[0]*size[1]];
    conv(frameArray, array, size, kernel, k, out);

    save_result(name, out, size);

    return 0;
}
