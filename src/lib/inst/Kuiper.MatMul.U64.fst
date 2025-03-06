module Kuiper.MatMul.U64

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"]
let kernel_u64 = kernel #u64

let matmul_u64 : matmul_ty u64 = matmul kernel_u64
