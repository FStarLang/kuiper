module Kuiper.MatMul.Naive2

(* This is a less naive matmul. It spawns full blocks of 1024 threads,
going in row major order through the output matrix and with each thread
computing a full dot product. *)

#lang-pulse

open Kuiper
open Kuiper.MatMulGPU.Type
open Kuiper.Matrix.Reprs.Type
module SZ = FStar.SizeT

inline_for_extraction noextract
val matmul_gpu
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA |}
  {| clayout lB |}
  {| clayout lC |}
  : matmul_gpu_ty_type_dims_repr et lA lB lC
