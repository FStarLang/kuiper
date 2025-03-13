module Kuiper.MatMul.Naive

(* This is a very naive matmul, spawning MxN blocks of 1 thread
to do the computation. *)

#lang-pulse

open Kuiper
open Kuiper.MatMulGPU.Type
open Kuiper.Matrix.Reprs.Type

inline_for_extraction noextract
val matmul_gpu : matmul_gpu_ty
  // (#et : Type0) {| scalar et |}
  // (#rows #shared #cols : szp)
  // (#lA : mlayout rows shared)
  // (#lB : mlayout shared cols)
  // (#lC : mlayout rows cols)
  // {| clayout lA |}
  // {| clayout lB |}
  // {| clayout lC |}
  // : matmul_gpu_ty_type_dims_repr et lA lB lC

