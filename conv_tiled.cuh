#define TW 16

__global__ void
conv_tiled( float* C, float* A, float* B, int wA, int hA, int wB)
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

  // kernel to shared memory
  // Faster: some threads load 2 all threads calculate

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
  int pad = 1; //std::floor(wB / 2); // Number of padding pixels

  // First tile row and column start from 0 others start interleaved 'pad' earlier
  size_t indA;
  if (i == 0)
    indA = i * wA + j;
  else
    indA = (i - pad) * wA + j;
  if (j > 0)
    indA -= pad;
  subTileA[ty][tx] = A[indA];

  // Sync so all data in subtile is present for calculations. Later there is no need 
  // for another synchronisation because the tiles are independent of each other
  __syncthreads();

  // Only calculate convolution for the non-padded area
  if ((i > pad) && (i < wA - pad) && (j > pad) && (j < hA - pad)) {
    // Calculate convolution. Separated to x and y for easier understanding
    for (int kx = 0; kx < wB; kx++) {
      for (int ky = 0; ky < wB; ky++) {
        accu += subTileA[ty - (ky - pad)][tx - (kx - pad)] * B[ky * wB + kx];
      }
    }
    // Copy result to output array (again keep only padded area)
    // Each thread writes one element
    C[(i - pad) * wA + j - pad] = accu;
  }
}

