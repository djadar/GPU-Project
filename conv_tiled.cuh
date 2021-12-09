#define TW 16

__global__ void
conv_tiled( float* C, float* A, float* B, int wA, int hA. int wB)
{
  /*
  Function that calculates the convolution between an array A and filter B.
  The convolution is done in tiles to save global memory access cost.
  Parameters:
    - C     Output filtered array
    - A     Padded input array
    - B     Used filter array
    - wA    Width of A
    - hA    Height of A
    - wB    Width of B

  */

  // Shared tile
  __shared__ float subTileA[TW][TW];
  
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int i = by * TW + ty;
  int j = bx * TW + tx;

  float accu = 0.0;
  int pad = std::floor(wB / 2) // Number of padding pixels

  subTileA[ty][tx] = A[i * wA + j];

  // Sync so all data in subtile is present for calculations. Later there is no need 
  // for another synchronisation because the tiles are independent of each other
  __syncthreads();

  // Only calculate convolution for the non-padded area
  if ((i > pad) && (i < wA - pad) && (j > pad) && (j < hA - pad)) {
    // Calculate convolution. Separated to x and y for easier understanding
    for (int kx = 0; kx < wB; kx++) {
      for (int ky = 0; ky < wB; ky++) {
        accu += subtileA[ty - (ky - pad)][tx - (kx - pad)] * B[ky * wB + kx];
      }
    }
    // Copy result to output array (again keep only padded area)
    // Each thread writes one element
    C[(i - pad) * wA + j - pad] = accu;
  }
}

