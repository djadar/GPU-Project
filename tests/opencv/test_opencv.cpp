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
    // std::cout << "" << std::endl;
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
    //
    //print_array("3- Input ", frameArray, size[1]);
    array_padding(frameArray, array, k/2, size[0], size[1]);
    //
    //print_array("4- Input", frameArray, size[1]);
    float total = 0;
    float elem = 0;
    int w_pad = size[1] + 2 * (k/2);
    // Go through each pixel in the original array
    for (int r = 0; r < size[0]; r++) {
        for (int c = 0; c < size[1]; c++) {
            total = 0;
            // Go through each element in the kernel array
            for (int x = 0; x < k; x++) {
                for (int y = 0; y < k; y++) {
                    elem = array[(r + y) * w_pad + c + x];
                    total += elem * kernel[x * k + y]; // Add to the total value for the output pixel
                }
            }
            out[r * size[1] + c] = total;
        }
    }
    //print_array("Output", out, size[0]);
}
int get_image(char* name, REAL *&frameArray, int *size ){

   //cv::Mat imgOriginal;        // input image
    cv::Mat imgGrayscale;        // grayscale of input image
   // print_array("2- Input", frameArray, size[1]);
    //imgOriginal
    cv::Mat image ;
    image = cv::imread(name);            // open image

    cv::cvtColor(image, imgGrayscale, CV_BGR2GRAY);        // convert to grayscale

    const int channels = image.channels();
    printf("Number of channels = %d \n", channels);

    
    imgGrayscale.copyTo(image); // Just to make sure the Mat objects are of the same size. 
    image = imgGrayscale;
    std::cout <<" \n" << std::endl;
  int numCols = size[1];
  int numRows = size[0];
  //std::cout  <<"numCols = " <<numCols <<" and numRows = " <<numRows <<" \n" << std::endl;
  //std::cout << "Output array" << std::endl;
  //frameArray = new REAL[numCols*numRows];
  REAL intensity;
  int value;
  for (int x = 0; x < numCols; x++) {          // x-axis, cols
    for (int y = 0; y < numRows; y++) { 
        intensity = frameArray[x * numCols + y];
        //std::cout << "voici"<<frameArray[x * numCols + y];
        if (intensity >= 0 && intensity <= 255){
            //value = floor(intensity);
            value = intensity;
        }else if (intensity < 0){
            value = 0;
        }else if (intensity > 255){
            value = 255;
        }
        //std::cout << frameArray[x * numCols + y];
        image.at<uchar>(x, y) = frameArray[x * numCols + y];
        //std::cout << "voici"<<image.at<uchar>(x, y);
    }
    // std::cout <<" \n" << std::endl;
  }
    std::cout <<" \n" << std::endl;
  //print_array("Input", frameArray, numCols);

  cv::namedWindow( "mes images", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Final", image );
  std::cout << "fini"<< std::endl;
  cv::waitKey(0);
}

int main( int argc, char** argv ) {
  
  
  char* name = "opencv_testimage.jpg";
  char* name2 = "bitmoji.png";
  
  int *size = (int *)malloc(sizeof(int)*2);
  //int n = get_image(name, frameArray, size);
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

  //std::cout  <<"numCols = " <<numCols <<" and numRows = " <<numRows <<" \n" << std::endl;
  //std::cout << "Output array" << std::endl;
  REAL *frameArray = new REAL[numCols*numRows];
  int intensity;
  for (int x = 0; x < numCols; x++) {          // x-axis, cols
    for (int y = 0; y < numRows; y++) {          // y-axis rows
        intensity = image.at<uchar>(cv::Point(y, x));
        //std::cout <<(double)image.at<uchar>(cv::Point(x, y)) <<" \n";
        frameArray[x * numRows + y] = intensity;
    }
  }

  //print_array("Input", frameArray, numCols);
  
  size[0] = numCols;
  size[1] = numRows;
  
  
  
    //kernel
    int k = 3;
    int pad = floor(k / 2);
    REAL *kernel = new REAL[k*k];
    sobel_filter(k, kernel);
    //print_array("Kernel",kernel, k);
    //Array
    REAL *array = new REAL[(size[0]+1)*(size[1]+1)];
    REAL *out = new REAL[size[0]*size[1]];
    conv(frameArray, array, size, kernel, k, out);
  
  int n = get_image(name, out, size);

  return 0;
}
