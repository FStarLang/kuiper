module Kuiper.MatMul.Tiled

#lang-pulse

(* TODO: Fit this into the non-tiled types. *)

open Kuiper
open Kuiper.MatMulGPU.Type
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix4 { mlayout4, clayout4 }

// inline_for_extraction noextract
// val matmul_gpu
//   (bdim : szp) (* block dim *)
//   (#et : Type0) {| scalar et |}
//   (#rows #shared #cols : szp) (* already divided by bdim *)
//   (_ : squash (bdim /? rows /\ bdim /? cols /\ bdim /? shared))
//   (lA : mlayout rows   shared)
//   (lB : mlayout shared cols)
//   (lC : mlayout rows   cols)
//   {| cA : clayout lA |}
//   {| cB : clayout lB |}
//   {| cC : clayout lC |}
//   : matmul_gpu_ty_type_dims_repr et #_ #rows #_ lA lB lC #cA #cB #cC
