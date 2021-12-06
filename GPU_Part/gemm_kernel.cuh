__global__ void
gemm_naive( float* C, float* A, float* B, int wA, int wB)
{
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Accumulate row i of A and column j of B
  int i = by * blockDim.y + ty;
  int j = bx * blockDim.x + tx;

  float accu = 0.0;

  for(int k=0; k<wA; k++){
    accu = accu + A[ i * wA + k ] * B[ k * wB + j ];
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  C[ i * wB + j ] = accu;

}

__global__ void
gemm_shared( float* C, float* A, float* B, int wA, int wB)
{
  
  __shared__ float subTileA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float subTileB[BLOCK_SIZE][BLOCK_SIZE];
  
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Accumulate row i of A and column j of B
  int i = by * BLOCK_SIZE + ty;
  int j = bx * BLOCK_SIZE + tx;

  float accu = 0.0;

  for(int k=0; k<wA/BLOCK_SIZE; k++){
    //accu = accu + A[ i * wA + k ] * B[ k * wB + j ];
    subTileA[ty][tx] = A[i*wA + k*BLOCK_SIZE+tx];
    subTileB[ty][tx] = B[(k * BLOCK_SIZE + ty)*wB + j ];
    __syncthreads();
    for(int m=0; m<BLOCK_SIZE; m++)
      accu = accu + subTileA[ty][m] * subTileB[m][tx];
    __syncthreads();  

  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  C[ i * wB + j ] = accu;

}