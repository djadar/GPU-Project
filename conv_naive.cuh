__global__ void
conv_naive( float* output, float* array, float* kernel, int k, int width)
{
  /* Naive function for calculating convolution between array and 
  * 2D kernel. In future requires tiling the image and the kernel 
  * into smaller pieces before calculating
  *
  * float* output:   output array
  * float* array:    padded input array
  * float* kernel:   the filter kernel
  * int k:           floor(width(kernel) / 2)
  * int width:       width of input array 
  */ 

  // thread indexing
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float accu = 0.0;
  // Go through each element in the filter kernel
  for(int r=0; r<k; r++){
    for(int c=0; c<k; c++){
        // start from (row-k, col-k) position and move through the
        // elements in the kernel
        accu = accu + array[(row - r - k)*width + (c - k)] * kernel[r*k + c]:
      }
  }
  // each thread writes one element to output matrix
  output[ row * width + col ] = accu;

}

