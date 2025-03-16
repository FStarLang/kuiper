module Kuiper.MatMul.Tiled

#lang-pulse
open Kuiper
open Kuiper.Poly.MatMulCPU { specialize_to_type_and_reprs as spec }
open Kuiper.Matrix.Reprs { row_major as RM, col_major as CM }
module P = Kuiper.Poly.MatMul.Tiled

let matmul_f32_tile16_rrr = spec (P.matmul_gpu 16sz) f32 RM RM RM
let matmul_f64_tile16_rrr = spec (P.matmul_gpu 16sz) f64 RM RM RM
let matmul_u32_tile16_rrr = spec (P.matmul_gpu 16sz) u32 RM RM RM
let matmul_u64_tile16_rrr = spec (P.matmul_gpu 16sz) u64 RM RM RM

let matmul_f32_tile16_ccc = spec (P.matmul_gpu 16sz) f32 CM CM CM
let matmul_f64_tile16_ccc = spec (P.matmul_gpu 16sz) f64 CM CM CM
let matmul_u32_tile16_ccc = spec (P.matmul_gpu 16sz) u32 CM CM CM
let matmul_u64_tile16_ccc = spec (P.matmul_gpu 16sz) u64 CM CM CM

let matmul_f32_tile32_rrr = spec (P.matmul_gpu 32sz) f32 RM RM RM
let matmul_f64_tile32_rrr = spec (P.matmul_gpu 32sz) f64 RM RM RM
let matmul_u32_tile32_rrr = spec (P.matmul_gpu 32sz) u32 RM RM RM
let matmul_u64_tile32_rrr = spec (P.matmul_gpu 32sz) u64 RM RM RM

let matmul_f32_tile32_ccc = spec (P.matmul_gpu 32sz) f32 CM CM CM
let matmul_f64_tile32_ccc = spec (P.matmul_gpu 32sz) f64 CM CM CM
let matmul_u32_tile32_ccc = spec (P.matmul_gpu 32sz) u32 CM CM CM
let matmul_u64_tile32_ccc = spec (P.matmul_gpu 32sz) u64 CM CM CM
