__global__
void ker(float *a, float *b, float *c, int m, int n, int k)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= m * n) return;
  int row = gid / n;
  int col = gid % n;
  float acc = 0.0f;
  float comp = 0.0f;
  for (int i = 0; i < k; i++) {
    float yc = a[row * k + i] * b[i * n + col] - comp;
    float t = acc + yc;
    comp = t - acc - yc;
    acc = t;
  }
  c[row * n + col] = acc;
}

void matmul(float *a, float *b, float *c, int m, int n, int k)
{
  int block_size = 256;
  int grid_size = (m * n + block_size - 1) / block_size;
  ker<<<grid_size, block_size>>>(a, b, c, m, n, k);
}
