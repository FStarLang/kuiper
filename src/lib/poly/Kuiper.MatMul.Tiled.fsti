module Kuiper.MatMul.Tiled

#lang-pulse

(* TODO: Fit this into the non-tiled types. *)

open Kuiper
open Kuiper.MatMulGPU.Type
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix4 { mlayout4, clayout4 }
module SZ = FStar.SizeT

inline_for_extraction
val kernel_fixed_ty
  (bdim : pos) (* block dim *)
  (et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : pos) (* already divided by bdim *)
  (lA : mlayout4 mrows   mshared bdim bdim)
  (lB : mlayout4 mshared mcols   bdim bdim)
  (lC : mlayout4 mrows   mcols   bdim bdim)
: Type0

inline_for_extraction noextract
val kernel_fixed
  (bdim : szp) (* block dim *)
  (#et : Type0) {| scalar et |}
  (#mrows #mshared #mcols : sz) (* already divided by bdim *)
  (#lA : mlayout4 mrows   mshared bdim bdim)
  (#lB : mlayout4 mshared mcols   bdim bdim)
  (#lC : mlayout4 mrows   mcols   bdim bdim)
  {| clayout4 lA |}
  {| clayout4 lB |}
  {| clayout4 lC |}
: kernel_fixed_ty bdim et lA lB lC

// inline_for_extraction noextract
// val matmul_gpu_fixed_wrap
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
//   (kk : kernel_fixed_ty bdim et lA lB lC #_ #_ #_)
//   : matmul_gpu_ty_type_dims_repr et #_ #rows #_ lA lB lC #cA #cB #cC
