module Klas.GEMM.TensorCore
#lang-pulse

open Kuiper
open Kuiper.Matrix.Reprs
open Klas.GEMM.TensorCore.Inst

let g_gemm_f16_f16_64x64x16_16x16x16 = specialize_gpu half half 64sz 64sz 16sz 16sz 16sz 16sz

let g_gemm_f16_f16_32x32x32_32x8x16 = specialize_gpu half half 32sz 32sz 32sz 32sz 8sz 16sz
let g_gemm_f16_f16_32x32x32_8x32x16 = specialize_gpu half half 32sz 32sz 32sz 8sz 32sz 16sz

let g_gemm_f16_f16_32x8x16_32x8x16 = specialize_gpu half half 32sz 8sz 16sz 32sz 8sz 16sz

let g_gemm_f16_f16_8x32x16_8x32x16 = specialize_gpu half half 32sz 32sz 32sz 8sz 32sz 16sz

// These instances are tested.
let g_gemm_f16_f16_64x64x64_16x16x16 = specialize_gpu half half 64sz 64sz 64sz 16sz 16sz 16sz
let g_gemm_f16_f16_64x64x64_32x8x16 = specialize_gpu half half 64sz 64sz 64sz 32sz 8sz 16sz
let g_gemm_f16_f16_64x64x64_8x32x16 = specialize_gpu half half 64sz 64sz 64sz 8sz 32sz 16sz

let g_gemm_f16_f16_32x32x32_16x16x16 = specialize_gpu half half 32sz 32sz 32sz 16sz 16sz 16sz

let g_gemm_f16_f16_16x16x16_16x16x16 = specialize_gpu half half 16sz 16sz 16sz 16sz 16sz 16sz

// mixed precision
let g_gemm_f16_f32_32x32x32_16x16x16 = specialize_gpu half float 32sz 32sz 32sz 16sz 16sz 16sz
