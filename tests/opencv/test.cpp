#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
 

int visit_image(char* name){

  cv::Mat image;
  image = cv::imread(name ,1);
  //cv::IMREAD_COLOR);
  if(! image.data ) {
      std::cout <<  "Image not found or unable to open" << std::endl ;
      return -1;
    }

  //produit convolutionnel
  std::vector<std::vector<int>> kernel = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
  //std::vector<std::vector<int>> out;
  std::vector<int> vect(image.rows-1, 10);
  std::vector<std::vector<int>> out(image.cols-1,vect);
  int total = 0;
  int elem = 0;
  std::cout << "Output array" << std::endl;

  for(int i = 1; i < image.rows -1; i++)
	{
	    std::cout << "[";
      for(int j = 1; j < image.cols -1; j++)
	    {
		      total = 0;
          for (int x = 0; x < kernel.size(); x++) {
              for (int y = 0; y < kernel[0].size(); y++) {
                  cv::Vec3b bgrPixel = image.at<cv::Vec3b>(i + x - 1,j + y - 1);
                  total +=  ((int)bgrPixel(0)) * kernel[x][y]; // Add to the total value for the output pixel
              }
          }
          out[i].push_back(total);
          //out[i][j] = total;
          std::cout << total << ", " ;
          //cv::Vec3b bgrPixel = image.at<cv::Vec3b>(i, j);
          
	    }
       std::cout << "]," << std::endl;
	}
  std::cout << std::endl;
  cv::namedWindow( "mes images", cv::WINDOW_AUTOSIZE );
  cv::imshow( "Initial", image );
  //cv::imshow( "Final", final );
  
  cv::waitKey(0);
}

int main( int argc, char** argv ) {
  
  
  char* name = "opencv_testimage.png";
  char* name2 = "bitmoji.png";
  int n = visit_image(name2);
  
  return n;
}
