#include <iostream>
#include <math.h>
#include <vector>

#define vec1d std::vector<int>
#define vec2d std::vector<std::vector<int>>

vec2d array_padding(int k, const vec2d &array) {
    size_t out_size = array.size() + 2*k;
    vec2d out(out_size);
    for (int r = 0; r < out_size; r++) {
        vec1d row(array.size() + 2*k, 0);
        if ((r > k) && (r < array.size() + k)) {
            for (int x = 0; x < array[0].size(); x++ ) {
                row[x+k] = array[r-k][x];
            }
        }
        out[r] = row;
    }
    return out;
}

int main() {
    vec2d array = {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {5, 2, 5, 2, 0}, {5, 2, 5, 2, 0}, {5, 2, 5, 2, 0}};
    vec2d kernel = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};;
    vec2d out = array;
    
    int kernel_size = 3; // hard coded for now
    int k = floor(kernel_size / 2);

    array = array_padding(k, array);

    int total = 0;
    int elem = 0;
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
    std::cout << "Output array" << std::endl;
    for (int i = 0; i < out.size(); i++) {
        std::cout << "[";
        for (int j = 0; j < out[0].size(); j++) {
            std::cout << out[i][j] << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
    return 0;
}