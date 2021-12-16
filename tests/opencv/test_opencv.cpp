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
    std::cout << "Matrix" <<name<<" \n[";
    for (int i = 0; i < M*M; i++) {
         
        std::cout << A[i] << " ";
        if ((i+1)%M ==0){
            std::cout <<"]\n";
            std::cout << "[";
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
void conv(){
    //
    array_padding(a_nopad, array, pad, 5, 5);
    //
    float total = 0;
    float elem = 0;
    int w_pad = w + 2 * pad;
    // Go through each pixel in the original array
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            total = 0;
            // Go through each element in the kernel array
            for (int x = 0; x < k; x++) {
                for (int y = 0; y < k; y++) {
                    elem = array[(r + y) * w_pad + c + x];
                    total += elem * kernel[x * k + y]; // Add to the total value for the output pixel
                }
            }
            out[r * w + c] = total;
        }
    }
}
int get_image(char* name, REAL *frameArray ){

  cv::Mat image;
  image = cv::imread(name ,1);
  //cv::IMREAD_COLOR);
  if(! image.data ) {
      std::cout <<  "Image not found or unable to open" << std::endl ;
      return -1;
    }

 cv::namedWindow( "mes images", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Initial", image );
  

  int numCols = image.cols;
  int numRows = image.rows;

  int size = numCols*numRows;

  int total = 0;
  int elem = 0;

  //std::cout << "Output array" << std::endl;
  frameArray = new REAL[size];

  for (int x = 0; x < numCols; x++) {          // x-axis, cols
    for (int y = 0; y < numRows; y++) {          // y-axis rows
        double intensity = image.at<uchar>(cv::Point(x, y));
        frameArray[x * numCols + y] = intensity;
    }
  }

  print_array("Input", frameArray, numCols);
  
  
  
  //cv::imshow( "Final", final );
  
  cv::waitKey(0);
}

int main( int argc, char** argv ) {
  
  
  char* name = "opencv_testimage.jpg";
  char* name2 = "bitmoji.png";
  REAL *frameArray ;

  int n = get_image(name, frameArray);
  
  //
  if n>0{
      //kernel
    int k = 3;
    int pad = floor(k / 2);
    sobel_filter(k, kernel);
    print_array(kernel, k);
    //Array
    REAL *array = 
    conv()
  }
  
  return n;
}
