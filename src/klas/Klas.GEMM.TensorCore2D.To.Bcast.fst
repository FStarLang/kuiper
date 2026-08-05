module Klas.GEMM.TensorCore2D.To.Bcast
#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Klas.GEMM.TensorCore2D.To.Bcast.Inst { spec }

let g_gemm_bcast_bf16_f32_bf16_64x64x16_16x16x16_2x2 = spec bf16 f32 bf16 64sz 64sz 16sz 16sz 16sz 16sz 2sz 2sz
