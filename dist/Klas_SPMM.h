
#ifndef Klas_SPMM_H
#define Klas_SPMM_H

#include <kuiper.h>

typedef struct Kuiper_Sparse_Matrix_smatrix__uint32_t_s {
    uint32_t nnz;
    uint32_t *elems;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_Matrix_smatrix__uint32_t;

void Klas_SPMM_spmm_u32(uint32_t rows, uint32_t shared, uint32_t cols,
                        Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                        uint32_t *row_indices, uint32_t *gB, uint32_t *gC);

typedef struct Kuiper_Sparse_Matrix_smatrix__float_s {
    uint32_t nnz;
    float *elems;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_Matrix_smatrix__float;

void Klas_SPMM_spmm_f32(uint32_t rows, uint32_t shared, uint32_t cols,
                        Kuiper_Sparse_Matrix_smatrix__float gA,
                        uint32_t *row_indices, float *gB, float *gC);

void Klas_SPMM_g_spmm_f32_32x4x1(uint32_t rows, uint32_t shared, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC);

void Klas_SPMM_g_spmm_f32_32x8x2(uint32_t rows, uint32_t shared, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC);

void Klas_SPMM_g_spmm_f32_32x16x4(uint32_t rows, uint32_t shared, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC);

void Klas_SPMM_g_spmm_f32_32x32x8(uint32_t rows, uint32_t shared, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC);

void Klas_SPMM_g_spmm_f32_32x64x8(uint32_t rows, uint32_t shared, uint32_t cols,
                                  Kuiper_Sparse_Matrix_smatrix__float gA,
                                  uint32_t *row_indices, float *gB, float *gC);

void Klas_SPMM_g_spmm_f32_32x4x1_on(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC,
                                    cudaStream_t s);

void Klas_SPMM_g_spmm_f32_32x8x2_on(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB, float *gC,
                                    cudaStream_t s);

void Klas_SPMM_g_spmm_f32_32x16x4_on(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_32x32x8_on(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_32x64x8_on(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_64x64x16(uint32_t rows, uint32_t shared,
                                   uint32_t cols,
                                   Kuiper_Sparse_Matrix_smatrix__float gA,
                                   uint32_t *row_indices, float *gB, float *gC);

void Klas_SPMM_g_spmm_f32_64x64x16_on(uint32_t rows, uint32_t shared,
                                      uint32_t cols,
                                      Kuiper_Sparse_Matrix_smatrix__float gA,
                                      uint32_t *row_indices, float *gB,
                                      float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_64x128x16(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB,
                                    float *gC);

void Klas_SPMM_g_spmm_f32_64x128x16_on(uint32_t rows, uint32_t shared,
                                       uint32_t cols,
                                       Kuiper_Sparse_Matrix_smatrix__float gA,
                                       uint32_t *row_indices, float *gB,
                                       float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_64x256x16(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB,
                                    float *gC);

void Klas_SPMM_g_spmm_f32_64x256x16_on(uint32_t rows, uint32_t shared,
                                       uint32_t cols,
                                       Kuiper_Sparse_Matrix_smatrix__float gA,
                                       uint32_t *row_indices, float *gB,
                                       float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_64x512x16(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB,
                                    float *gC);

void Klas_SPMM_g_spmm_f32_64x512x16_on(uint32_t rows, uint32_t shared,
                                       uint32_t cols,
                                       Kuiper_Sparse_Matrix_smatrix__float gA,
                                       uint32_t *row_indices, float *gB,
                                       float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_128x64x16(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB,
                                    float *gC);

void Klas_SPMM_g_spmm_f32_128x64x16_on(uint32_t rows, uint32_t shared,
                                       uint32_t cols,
                                       Kuiper_Sparse_Matrix_smatrix__float gA,
                                       uint32_t *row_indices, float *gB,
                                       float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_128x128x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_128x128x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_128x128x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_128x128x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_128x256x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_128x256x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_128x256x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_128x256x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_128x512x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_128x512x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_128x512x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_128x512x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_256x64x16(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB,
                                    float *gC);

void Klas_SPMM_g_spmm_f32_256x64x16_on(uint32_t rows, uint32_t shared,
                                       uint32_t cols,
                                       Kuiper_Sparse_Matrix_smatrix__float gA,
                                       uint32_t *row_indices, float *gB,
                                       float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_256x128x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_256x128x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_256x128x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_256x128x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_256x256x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_256x256x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_256x256x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_256x256x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_256x256x64(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_256x256x64_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_256x512x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_256x512x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_256x512x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_256x512x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_256x512x64(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_256x512x64_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_512x64x16(uint32_t rows, uint32_t shared,
                                    uint32_t cols,
                                    Kuiper_Sparse_Matrix_smatrix__float gA,
                                    uint32_t *row_indices, float *gB,
                                    float *gC);

void Klas_SPMM_g_spmm_f32_512x64x16_on(uint32_t rows, uint32_t shared,
                                       uint32_t cols,
                                       Kuiper_Sparse_Matrix_smatrix__float gA,
                                       uint32_t *row_indices, float *gB,
                                       float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_512x128x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_512x128x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_512x128x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_512x128x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_512x256x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_512x256x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_512x256x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_512x256x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_512x256x64(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_512x256x64_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_512x512x16(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_512x512x16_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_512x512x32(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_512x512x32_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_512x512x64(uint32_t rows, uint32_t shared,
                                     uint32_t cols,
                                     Kuiper_Sparse_Matrix_smatrix__float gA,
                                     uint32_t *row_indices, float *gB,
                                     float *gC);

void Klas_SPMM_g_spmm_f32_512x512x64_on(uint32_t rows, uint32_t shared,
                                        uint32_t cols,
                                        Kuiper_Sparse_Matrix_smatrix__float gA,
                                        uint32_t *row_indices, float *gB,
                                        float *gC, cudaStream_t s);

void Klas_SPMM_g_spmm_f32_512x512x128(uint32_t rows, uint32_t shared,
                                      uint32_t cols,
                                      Kuiper_Sparse_Matrix_smatrix__float gA,
                                      uint32_t *row_indices, float *gB,
                                      float *gC);

void Klas_SPMM_g_spmm_f32_512x512x128_on(uint32_t rows, uint32_t shared,
                                         uint32_t cols,
                                         Kuiper_Sparse_Matrix_smatrix__float gA,
                                         uint32_t *row_indices, float *gB,
                                         float *gC, cudaStream_t s);

void Klas_SPMM_spmm_f32_dispatch(uint32_t rows, uint32_t shared, uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t *row_indices, float *gB, float *gC);

#define Klas_SPMM_H_DEFINED
#endif /* Klas_SPMM_H */
