
#ifndef Klas_SPMM_H
#define Klas_SPMM_H

#include <kuiper.h>

typedef struct Kuiper_Sparse_Matrix_smatrix__uint32_t_s {
    uint32_t nnz;
    uint32_t *elems;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_Matrix_smatrix__uint32_t;

void
Klas_SPMM_spmm_u32(uint32_t rows,
                   uint32_t shared,
                   uint32_t cols,
                   Kuiper_Sparse_Matrix_smatrix__uint32_t gA,
                   uint32_t * row_indices, uint32_t * gB, uint32_t * gC);

typedef struct Kuiper_Sparse_Matrix_smatrix__float_s {
    uint32_t nnz;
    float *elems;
    uint32_t *col_ind;
    uint32_t *row_off;
} Kuiper_Sparse_Matrix_smatrix__float;

void
Klas_SPMM_g_spmm_f32_16x16x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_16x32x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_16x64x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_16x128x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_16x256x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_16x512x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_32x16x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_32x32x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_32x32x32(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_32x64x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_32x64x32(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_32x128x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_32x128x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_32x256x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_32x256x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_32x512x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_32x512x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x16x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x32x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x32x32(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x64x16(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x64x32(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x64x64(uint32_t rows,
                              uint32_t shared,
                              uint32_t cols,
                              Kuiper_Sparse_Matrix_smatrix__float gA,
                              uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x128x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x128x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x128x64(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x256x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x256x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x256x64(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x512x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x512x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_64x512x64(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x16x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x32x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x32x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x64x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x64x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x64x64(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x128x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x128x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x128x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x128x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x256x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x256x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x256x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x256x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x512x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x512x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x512x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_128x512x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x16x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x32x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x32x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x64x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x64x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x64x64(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x128x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x128x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x128x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x128x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x256x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x256x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x256x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x256x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x256x256(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x512x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x512x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x512x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x512x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_256x512x256(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x16x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x32x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x32x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x64x16(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x64x32(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x64x64(uint32_t rows,
                               uint32_t shared,
                               uint32_t cols,
                               Kuiper_Sparse_Matrix_smatrix__float gA,
                               uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x128x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x128x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x128x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x128x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x256x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x256x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x256x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x256x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x256x256(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x512x16(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x512x32(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x512x64(uint32_t rows,
                                uint32_t shared,
                                uint32_t cols,
                                Kuiper_Sparse_Matrix_smatrix__float gA,
                                uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x512x128(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x512x256(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

void
Klas_SPMM_g_spmm_f32_512x512x512(uint32_t rows,
                                 uint32_t shared,
                                 uint32_t cols,
                                 Kuiper_Sparse_Matrix_smatrix__float gA,
                                 uint32_t * row_indices, float *gB, float *gC);

#define Klas_SPMM_H_DEFINED
#endif                          /* Klas_SPMM_H */
