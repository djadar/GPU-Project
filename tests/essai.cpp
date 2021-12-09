#include <iostream>
#include <cstdlib>
//#include <opencv2/cv.h>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/highgui.h>

int main(int argc, char**argv){

  cv::Mat img = cv::imread("opencv_testimage.png",1);
  cv::imshow("image",img);
  cv::waitKey(0);
  cv::Vec3b firstline[img.cols];
  for(int i=0;i<img.cols;i++){
    // access to matrix
    cv::Vec3b tmp = img.at<cv::Vec3b>(0,i);
    std::cout << (int)tmp(0) << " " << (int)tmp(1) << " " << (int)tmp(2) << std::endl;
    // access to my array
    firstline[i] = tmp;
    std::cout << (int)firstline[i](0) << " " << (int)firstline[i](0) << " " << (int)firstline[i](0) << std::endl;
  }
  return EXIT_SUCCESS;
}
