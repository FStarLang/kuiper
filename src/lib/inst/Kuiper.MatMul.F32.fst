module Kuiper.MatMul.F32

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"]
let kernel_f32 = kernel #f32

let matmul_f32 : matmul_ty f32 = matmul kernel_f32
