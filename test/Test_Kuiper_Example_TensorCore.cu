#include "test-common.h"

// Huge hack: including the cu here. The right thing is writing
// an actual kernel and call it in the example file. This requires
// the device function there to be inline, so nvcc won't complain
// about multiple declarations.
#include "Kuiper_Example_TensorCore.cu"

__global__
void k(half_t *a, half_t *b, half_t *c)
{
    Kuiper_Example_TensorCore_test(a, b, c);
}

int main ()
{
        half a[16][16];
        half b[16][16];
        half c[16][16];
        half c_check[16][16];

        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                a[i][j] = __float2half((float)(i + j) / 4);
                b[i][j] = __float2half((float)(i - j) / 4);
                c[i][j] = __float2half(0.0f);
            }
        }

        half *ga = nullptr;
        half *gb = nullptr;
        half *gc = nullptr;

        cudaMalloc(&ga, sizeof(half) * 16 * 16);
        cudaMalloc(&gb, sizeof(half) * 16 * 16);
        cudaMalloc(&gc, sizeof(half) * 16 * 16);

        cudaMemcpy(ga, a, sizeof(half) * 16 * 16, cudaMemcpyHostToDevice);
        cudaMemcpy(gb, b, sizeof(half) * 16 * 16, cudaMemcpyHostToDevice);
        cudaMemcpy(gc, c, sizeof(half) * 16 * 16, cudaMemcpyHostToDevice);

        k<<<1,32>>>(ga, gb, gc);

        cudaMemcpy(c, gc, sizeof(half) * 16 * 16, cudaMemcpyDeviceToHost);

        // compute c_check in the CPU
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                c_check[i][j] = __float2half(0.0f);
                for (int k = 0; k < 16; k++) {
                    c_check[i][j] = __hadd(c_check[i][j], __hmul(a[i][k], b[k][j]));
                }
            }
        }

        // compare c and c_check
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                float diff = c[i][j] - c_check[i][j];
                float rel = fabs (diff / __half2float(c[i][j]));
                /* check for equality just in case they are both zero */
                bool ok = (c[i][j] == c_check[i][j]) || rel <= 0.1;
                if (! ok) {
                    printf("Mismatch at (%d, %d): GPU = %g, CPU = %g, relative error = %g\n",
                           i, j, __half2float(c[i][j]), __half2float(c_check[i][j]), rel);
                }
            }
        }

        printf("Tensor core ok!\n");

        cudaFree(ga);
        cudaFree(gb);
        cudaFree(gc);

        return 0;
}
