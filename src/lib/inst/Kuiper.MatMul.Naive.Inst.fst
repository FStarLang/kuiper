module Kuiper.MatMul.Naive.Inst

#lang-pulse
open Kuiper
open Kuiper.MatMul.Naive
open Kuiper.MatMulCPU { specialize_to_type_and_reprs as spec }
open Kuiper.MatMulGPU.Type
open Kuiper.Matrix.Reprs { row_major as R, col_major as C }

let matmul_f32_rrr = spec matmul_gpu f32 R R R
let matmul_f64_rrr = spec matmul_gpu f64 R R R
let matmul_u32_rrr = spec matmul_gpu u32 R R R
let matmul_u64_rrr = spec matmul_gpu u64 R R R

let matmul_f32_ccc = spec matmul_gpu f32 C C C
let matmul_f64_ccc = spec matmul_gpu f64 C C C
let matmul_u32_ccc = spec matmul_gpu u32 C C C
let matmul_u64_ccc = spec matmul_gpu u64 C C C


