module Kuiper.MatMul.U32

#lang-pulse
open Kuiper
open Kuiper.MatMul

[@@CPrologue "__global__"]
let kernel_u32 = kernel #u32

let matmul_u32 : matmul_ty u32 = matmul kernel_u32
