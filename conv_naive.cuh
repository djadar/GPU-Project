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

  int pad = (k + 1) / 2;
  int w_pad = width + pad * 2
  // thread indexing
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float accu = 0.0;
  // Go through each element in the filter kernel
  for(int x=0; x<k; x++){
    for(int y=0; y<k; y++){
        // start from (row-k, col-k) position and move through the
        // elements in the kernel
        accu = accu + array[(row + y)*w_pad + col + x] * kernel[y * k + x];
      }
  }
  // each thread writes one element to output matrix
  if ((row >= 0) && (row < width) && (col >= 0) && (col < width)) {
    output[ row * width + col ] = accu;
    //printf("(%d, %d): %f \n", row, col, accu);
    //output[ row * width + col] = 1.0;
  }
}
