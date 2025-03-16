module Kuiper.Poly.MatMul.Naive2

(* This is a less naive matmul. It spawns full blocks of 1024 threads,
going in row major order through the output matrix and with each thread
computing a full dot product. *)

#lang-pulse

open Kuiper
open Kuiper.Poly.MatMulGPU.Type

inline_for_extraction noextract
val matmul_gpu : matmul_gpu_ty
