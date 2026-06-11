#define tile 16
#include <cassert>

__global__
void ker(float *A, float *B, float *C, int m, int n, int k)
{
  __shared__ float cA[tile * tile];
  __shared__ float cB[tile * tile];
  int tile_r = blockIdx.x / tile;
  int tile_c = blockIdx.x % tile;
  int sr = threadIdx.x / tile;
  int sc = threadIdx.y % tile;

  int r = sr + tile_r * tile;
  int c = sc + tile_c * tile;

  float sum = 0.0f;
  for (int kk = 0; kk < k; kk += tile) {
    __syncthreads();
    cA[sr * tile + sc] = A[kk * k + sr * k + sc];
    cB[sr * tile + sc] = B[kk * n + sr * n + sc];
    __syncthreads();

    for (int k = 0; k < tile; k++)
      sum += cA[sr * tile + k] * cB[sr * tile + k];
  }
  C[r * n + c] = sum;
}

void matmul(float *a, float *b, float *c, int m, int n, int k)
{
  assert (m > 0 && n > 0 && k > 0);
  assert (m % tile == 0);
  assert (n % tile == 0);
  assert (k % tile == 0);
  assert (tile * tile <= 1024);
  int grid_size = (m / tile) * (n / tile);
  ker<<<grid_size, tile*tile>>>(a, b, c, m, n, k);
}
