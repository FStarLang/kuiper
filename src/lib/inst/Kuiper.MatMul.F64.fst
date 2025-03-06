module Kuiper.MatMul.F64

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"]
let kernel_f64 = kernel #f64

let matmul_f64 : matmul_ty f64 = matmul kernel_f64
