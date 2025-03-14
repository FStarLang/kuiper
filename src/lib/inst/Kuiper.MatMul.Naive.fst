module Kuiper.MatMul.Naive

#lang-pulse
open Kuiper
open Kuiper.Poly.MatMulCPU { specialize_to_type_and_reprs as spec }
open Kuiper.Poly.MatMulGPU.Type
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }
module P = Kuiper.Poly.MatMul.Naive

let matmul_f32_rrr = spec P.matmul_gpu f32 RM RM RM
let matmul_f64_rrr = spec P.matmul_gpu f64 RM RM RM
let matmul_u32_rrr = spec P.matmul_gpu u32 RM RM RM
let matmul_u64_rrr = spec P.matmul_gpu u64 RM RM RM

let matmul_f32_ccc = spec P.matmul_gpu f32 CM CM CM
let matmul_f64_ccc = spec P.matmul_gpu f64 CM CM CM
let matmul_u32_ccc = spec P.matmul_gpu u32 CM CM CM
let matmul_u64_ccc = spec P.matmul_gpu u64 CM CM CM
