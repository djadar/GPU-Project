__global__ void
conv_naive( float* out, float* A, float* K, int wK, int wA, int hA)
{
  /* Naive function for calculating convolution between array and 
  * 2D kernel. In future requires tiling the image and the kernel 
  * into smaller pieces before calculating
  *
  * float* out:   output array
  * float* A:     padded input array A
  * float* K:     the filter kernel array
  * int wK:       width of the filter kernel
  * int wA:       width of the input array
  * int hA:       height of the input array
  */

  int pad = wK / 2;
  int w_pad = wA + pad * 2;
  // thread indexing
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float accu = 0.0;
  // Go through each element in the filter kernel
  for(int y=0; y<wK; y++){
    for(int x=0; x<wK; x++){
        // start from (row-k, col-k) position and move through the
        // elements in the kernel
        accu = accu + A[(row + y)*w_pad + col + x] * K[y * wK + x];
      }
  }
  // each thread writes one element to output matrix
  if ((row < hA) && (col < wA)) {
    out[ row * wA + col ] = accu;
    //printf("(%d, %d): %f \n", row, col, accu);
  }
}
