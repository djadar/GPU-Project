# GPU-Project
Application of parallel programming with Cuda

# Repository contents

### Main code
* `main_cpu.cxx` - Contains the CPU main code and convolution function
* `main_gpu.cu`  - Contains the GPU main code 

### GPU kernels
* `conv_naive.cuh` - Naive convolution function without tiling 
* `conv_tiled.cuh` - Tiled convolutional function
* `conv_shared.cuh` - Tiled convolutional function with shared memory filter

### Utilities
* `utils.h` - Utility functions used by the main programs for data manipulation and printing
* `image_utils.h` - Utility functions for manipulating an image using Opencv
