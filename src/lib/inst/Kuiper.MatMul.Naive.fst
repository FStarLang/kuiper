module Kuiper.MatMul.Naive

#lang-pulse
open Kuiper
open Kuiper.Poly.MatMulCPU {
  matmul_cpu,
  specialize_to_type_and_reprs as spec
}
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }
module P = Kuiper.Poly.MatMul.Naive

let matmul_f32_rrr = spec (matmul_cpu P.kdesc) f32 RM RM RM
let matmul_f64_rrr = spec (matmul_cpu P.kdesc) f64 RM RM RM
let matmul_u32_rrr = spec (matmul_cpu P.kdesc) u32 RM RM RM
let matmul_u64_rrr = spec (matmul_cpu P.kdesc) u64 RM RM RM

let matmul_f32_ccc = spec (matmul_cpu P.kdesc) f32 CM CM CM
let matmul_f64_ccc = spec (matmul_cpu P.kdesc) f64 CM CM CM
let matmul_u32_ccc = spec (matmul_cpu P.kdesc) u32 CM CM CM
let matmul_u64_ccc = spec (matmul_cpu P.kdesc) u64 CM CM CM
