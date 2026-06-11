__global__
void ker(float *a, float *b, float *c, int m, int n, int k)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= m * n) return;
  int row = gid / n;
  int col = gid % n;
  float sum = 0.0f;
  for (int i = k-1; i >= 0; i--)
    sum += a[row * k + i] * b[i * n + col];
  c[row * n + col] = sum;
}

void matmul(float *a, float *b, float *c, int m, int n, int k)
{
  int block_size = 256;
  int grid_size = (m * n + block_size - 1) / block_size;
  ker<<<grid_size, block_size>>>(a, b, c, m, n, k);
}
