module Kuiper.MatMul.Naive

#lang-pulse

open Kuiper
open Kuiper.MatMulGPU.Type
open Kuiper.Matrix.Reprs.Type
module SZ = FStar.SizeT

inline_for_extraction
val kernel_fixed_ty
  (et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (lA : mlayout rows shared)
  (lB : mlayout shared cols)
  (lC : mlayout rows cols)
  {| clayout lA |}
  {| clayout lB |}
  {| clayout lC |}
: Type0

inline_for_extraction noextract
val kernel_fixed
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : SZ.t)
  (lA : mlayout rows shared)
  (lB : mlayout shared cols)
  (lC : mlayout rows cols)
  {| clayout lA |}
  {| clayout lB |}
  {| clayout lC |}
: kernel_fixed_ty et lA lB lC

inline_for_extraction noextract
val matmul_gpu_fixed
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : szp)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lA |}
  {| clayout lB |}
  {| clayout lC |}
  (kk : kernel_fixed_ty et lA lB lC #_ #_ #_)
  : matmul_gpu_ty_type_dims_repr et lA lB lC
