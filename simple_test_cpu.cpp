#include <iostream>
#include <vector>

int main() {
    std::cout << "jaa" << std::endl;
    
    std::vector<std::vector<int>> array = {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {5, 2, 5, 2, 0}, {5, 2, 5, 2, 0}, {5, 2, 5, 2, 0}};
    std::vector<std::vector<int>> kernel = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    std::vector<std::vector<int>> out = array;

    int total = 0;
    int elem = 0;
    std::cout << "Output array" << std::endl;
    // Go through each pixel in the original array
    for (int i = 1; i < array.size() - 1; i++) {
        std::cout << "[";
        for (int j = 1; j < array[0].size() - 1; j++) {
            total = 0;
            // Go through each element in the kernel array
            for (int x = 0; x < kernel.size(); x++) {
                for (int y = 0; y < kernel[0].size(); y++) {
                    elem = array[i + x - 1][j + y - 1];
                    total +=  elem * kernel[x][y]; // Add to the total value for the output pixel
                }
            }
            out[i][j] = total;
            std::cout << total << ", " ;
        }
        std::cout << "]," << std::endl;
    }
    std::cout << std::endl;
    return 0;
}