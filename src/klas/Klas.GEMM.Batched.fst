module Klas.GEMM.Batched

#lang-pulse
open Kuiper
module K = Kuiper.Kernel.BatchedGEMM

let batched_gemm_f32 = K.batched_gemm_f32
