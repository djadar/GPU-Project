#include <iostream>
#include <math.h>
#include <vector>
//#include <opencv2/opencv.hpp>

#define vec1df std::vector<float>
#define vec2df std::vector<std::vector<float>>

// Create sobel filter with size k x k
vec2df sobel_filter(int k) {
    vec2df out(k);
    float v, x_dist, y_dist;
    for (int i = 0; i < k; i++) {
        vec1df row(k);
        for (int j = 0; j < k; j++) {
            if (j == floor(k/2)){
                v = 0;
            }
            else {
                y_dist = (i - floor(k/2));
                x_dist = (j - floor(k/2));
                v = x_dist / (x_dist * x_dist + y_dist * y_dist);
            }
            row[j] = v;
        }
        out[i] = row;
    }
    return out;
}

// Create padded input vector padded by k pixels
vec2df array_padding(int k, const vec2df &array) {
    size_t out_size = array.size() + 2*k;
    vec2df out(out_size);
    for (int r = 0; r < out_size; r++) {
        vec1df row(array.size() + 2*k, 0);
        if ((r >= k) && (r < array.size() + k)) {
            for (int x = 0; x < array[0].size(); x++ ) {
                row[x+k] = array[r-k][x];
            }
        }
        out[r] = row;
    }
    return out;
}

void print_array(const vec2df &X) {
    for (int i = 0; i < X.size(); i++) {
        std::cout << "[";
        for (int j = 0; j < X[0].size(); j++) {
            std::cout << X[i][j] << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

// TODO: Add arguments (image path, kernel size)
int main() {
    int kernel_size = 3;
    int k = floor(kernel_size / 2);
    // TODO: Replace with opening an image with opencv
    vec2df array = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    vec2df out = array;
    array = array_padding(k, array);
    
    vec2df kernel = sobel_filter(kernel_size);
    std::cout << "Kernel" << std::endl;
    print_array(kernel);

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

    return 0;
}