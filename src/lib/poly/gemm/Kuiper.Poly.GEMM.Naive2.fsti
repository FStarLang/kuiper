module Kuiper.Poly.GEMM.Naive2

(* This is a less naive matmul. It spawns full blocks of 1024 threads,
going in row major order through the output matrix and with each thread
computing a full dot product. *)

#lang-pulse

open Kuiper

open Kuiper.Poly.GEMMGPU.Type

inline_for_extraction noextract
let size_req : size_req_t =
  fun m n k -> m * n <= max_blocks * max_threads

inline_for_extraction noextract
val mmcomb_gpu_exact : matmulcomb_gpu_ty size_req

inline_for_extraction noextract
val mmcomb_gpu_approx : matmulcomb_gpu_approx_ty size_req
