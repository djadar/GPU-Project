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

void fillA(REAL *&A, int M) {
    //REAL array [M*M] = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 2, 5, 2, 0, 5, 2, 5, 2, 0, 5, 2, 5, 2, 0};
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
        A[i * M + j] = 1;
        //array[i * M + j];
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

int get_image(char* name){

  cv::Mat image;
  image = cv::imread(name ,1);
  //cv::IMREAD_COLOR);
  if(! image.data ) {
      std::cout <<  "Image not found or unable to open" << std::endl ;
      return -1;
    }

 cv::namedWindow( "mes images", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Initial", image );
  //produit convolutionnel
  std::vector<std::vector<int>> kernel = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
  //std::vector<std::vector<int>> out;
  //std::vector<int> vect(image.rows-1, 10);
  //std::vector<std::vector<int>> out(image.cols-1,vect);

  int numCols = image.cols;
  int numRows = image.rows;

  int size = numCols*numRows;

  int total = 0;
  int elem = 0;

  std::cout << "Output array" << std::endl;
  REAL *frameArray = new REAL[size];

  for (int x = 0; x < numCols; x++) {          // x-axis, cols
    for (int y = 0; y < numRows; y++) {          // y-axis rows
        double intensity = image.at<uchar>(Point(x, y));
        frameArray[x * numCols + y] = intensity;
    }
  }

  print_array("Input", frameArray, size);
  //cv::namedWindow( "mes images", cv::WINDOW_AUTOSIZE );
  //cv::imshow( "Initial", image );
  //cv::imshow( "Final", final );
  
  cv::waitKey(0);
}

int main( int argc, char** argv ) {
  
  
  char* name = "opencv_testimage.jpg";
  char* name2 = "bitmoji.png";
  int n = get_image(name2);
  
  return n;
}
