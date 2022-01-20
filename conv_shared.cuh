#define TW 16
#define WIDTH_K 3

#define REAL float
__global__ void
conv_shared( REAL* out, REAL* A, REAL* K, int wA, int hA)
{
  /*
  Function that calculates the convolution between an array array and filter B.
  The convolution is done in tiles to save global memory access cost.
  Parameters:
    - float* out    Output filtered array
    - float* A      Padded input array
    - float* K      Used filter array
    - int wA        Width of the non-padded input array
    - int hA        Height of the non-padded input array
  */
  // Shared tile and filter kernel array
  __shared__ float subTile[TW][TW];
  __shared__ float kernel[WIDTH_K][WIDTH_K];
  
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int i = by * TW + ty;
  int j = bx * TW + tx;

  float accu = 0.0;
  int pad = WIDTH_K / 2; // Kernel has to always be odd numbered

  // row and col moving forward TW - 2*pad indices per tile
  //   by * 2*pad removed from row to get current tile's row index
  //   bx * 2*pad removed from col to get current tile's col index
  //     -> row index * width + col index results in correct indexing for the subtile
  size_t ind = (i - by * 2*pad) * (wA + 2*pad) + j - bx * 2*pad;
  subTile[ty][tx] = A[ind];
  if ((ty < WIDTH_K) && (tx < WIDTH_K))
    kernel[ty][tx] = K[ty * WIDTH_K + tx];

  // Sync so all data in subtile is present for calculations. Later there is no need 
  // for another synchronisation because the tiles are independent of each other
  __syncthreads();

  for(int y=0; y<WIDTH_K; y++){
    for(int x=0; x<WIDTH_K; x++){
      // start from (row-k, col-k) position and move through the
      // elements in the kernel
      accu += subTile[ty + y][tx + x] * kernel[y][x];
    }
  }
  // Only the center elements of convolution are needed, skip padded area around calculation
  if ((ty < TW - 2*pad) && (tx < TW - 2*pad)) {
    int outy = (by * (TW - 2*pad) + ty);
    int outx = bx * (TW - 2*pad) + tx;
    // Do not write elements that are outside of output bounds because of tiling
    if ((outx < wA) && (outy < hA)) {
      // Index calculation:
      //   Each row and column of a tile increases the index in that direction by 2*pad
      //   So add multiple of that to each row + the thread row and to each column + the thread column 
      out[outy * wA + outx] = accu;
    }
  }
}
