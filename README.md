# GPU-Project
Application of parallel programming with Cuda

**1. Repository contents**

#### Main code
* `main_cpu.cxx` - Contains the CPU main code and convolution function
* `main_gpu.cu`  - Contains the GPU main code. 

#### GPU kernels
* `conv_naive.cuh` - Naive convolution function without tiling 
* `conv_shared.cuh` - Tiled convolutional function with shared memory filter.

#### Utilities
* `utils.h` - Utility functions used by the main programs for data manipulation and printing
* `image_utils.h` - Utility functions for manipulating an image using Opencv.


**2.  How to compile the project the first time**

`mkdir build && cd build`

`cmake ..`

`make`

Remark : make sure to be on a gpu environnement to compile the `main_gpu.cu` file.

**3.  How to test**

a) CPU version

Run those commands :

`cd build`

`./edge_cpu` with these options :

OPTIONS:

      -h, --help                        Display this help menu
      --WC=[widthC]                     Width of output matrix C
      --HC=[heightC]                    Height of output matrix C
      --WK=[widthB]                     Width of kernel matrix K

The defaut parameters are : WC = 256, HC = 256 and WK = 3.

b) GPU version

Run those commands :

`cd build`

`./edge_cuda` with these options :

OPTIONS:

      -h, --help                        Display this help menu
      --WC=[widthC]                     Width of output matrix C
      --HC=[heightC]                    Height of output matrix C
      --WK=[widthB]                     Width of kernel matrix K
      --choice=[choice]                 Choose the way of doing the calculation
                                        1 : conv_naive ; 2: conv_tiled_shared
The defaut parameters are : WC = 256, HC = 256, WK = 3 and choice = 1

**4. To get the time**

perl extract.pl < data.txt > data.csv

