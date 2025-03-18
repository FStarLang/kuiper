module Kuiper.GEMM.Naive

#lang-pulse
open Kuiper
open Kuiper.Poly.GEMMCPU {
  specialize_as_matmul_to_type_and_reprs_cpu as spec_cpu,
  specialize_as_matmul_to_type_and_reprs_gpu as spec_gpu
}
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }
module P = Kuiper.Poly.GEMM.Naive

let matmul_f32_rrr = spec_cpu P.matmul_gpu f32 RM RM RM
let matmul_f64_rrr = spec_cpu P.matmul_gpu f64 RM RM RM
let matmul_u32_rrr = spec_cpu P.matmul_gpu u32 RM RM RM
let matmul_u64_rrr = spec_cpu P.matmul_gpu u64 RM RM RM

let matmul_f32_ccc = spec_cpu P.matmul_gpu f32 CM CM CM
let matmul_f64_ccc = spec_cpu P.matmul_gpu f64 CM CM CM
let matmul_u32_ccc = spec_cpu P.matmul_gpu u32 CM CM CM
let matmul_u64_ccc = spec_cpu P.matmul_gpu u64 CM CM CM

let g_matmul_f32_rrr = spec_gpu P.matmul_gpu f32 RM RM RM
let g_matmul_f64_rrr = spec_gpu P.matmul_gpu f64 RM RM RM
let g_matmul_u32_rrr = spec_gpu P.matmul_gpu u32 RM RM RM
let g_matmul_u64_rrr = spec_gpu P.matmul_gpu u64 RM RM RM

let g_matmul_f32_ccc = spec_gpu P.matmul_gpu f32 CM CM CM
let g_matmul_f64_ccc = spec_gpu P.matmul_gpu f64 CM CM CM
let g_matmul_u32_ccc = spec_gpu P.matmul_gpu u32 CM CM CM
let g_matmul_u64_ccc = spec_gpu P.matmul_gpu u64 CM CM CM
