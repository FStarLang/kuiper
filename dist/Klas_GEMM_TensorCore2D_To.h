
#ifndef Klas_GEMM_TensorCore2D_To_H
#define Klas_GEMM_TensorCore2D_To_H

#include <kuiper.h>

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x16_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x32_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x64x64_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_2x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x16_16x16x16_4x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_2x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x32_16x16x16_4x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_2x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_64x128x64_16x16x16_4x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x16_16x16x16_8x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x32_16x16x16_8x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x64x64_16x16x16_8x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_2x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_4x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x16_16x16x16_8x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_2x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_4x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x32_16x16x16_8x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_2x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_4x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x2(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x4(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

void Klas_GEMM_TensorCore2D_To_g_gemm_bf16_f32_bf16_128x128x64_16x16x16_8x8(
    uint32_t rows, uint32_t shared, uint32_t cols, __nv_bfloat16 *gA,
    __nv_bfloat16 *gB, __nv_bfloat16 *gC, __nv_bfloat16 *gD, float alpha,
    float beta);

#define Klas_GEMM_TensorCore2D_To_H_DEFINED
#endif /* Klas_GEMM_TensorCore2D_To_H */
