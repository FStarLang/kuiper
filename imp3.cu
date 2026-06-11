#define tile 16

__global__
void ker(float *a, float *b, float *c, int m, int n, int k)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= m * n) return;
  int row = gid / n;
  int col = gid % n;
  float sum = 0.0f;
  for (int k0 = 0; k0 < k; k0 += tile) {
    float acc = 0.0f;
    for (int k1 = 0; k1 < tile && k0 + k1 < k; k1++)
      acc += a[row * k + k0 + k1] * b[k0 + k1 * n + col];
    sum += acc;
  }
  c[row * n + col] = sum;
}

void matmul(float *a, float *b, float *c, int m, int n, int k)
{
  int block_size = 256;
  int grid_size = (m * n + block_size - 1) / block_size;
  ker<<<grid_size, block_size>>>(a, b, c, m, n, k);
}
