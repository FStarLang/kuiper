module Kuiper.MatMul

#lang-pulse

open Kuiper
module M = Kuiper.Matrix
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT
open Kuiper.EMatrix

inline_for_extraction
val kernel_ty (et : Type0) {| scalar et |} : Type0

inline_for_extraction noextract
val kernel (#et : Type0) {| scalar et |} : kernel_ty et

unfold
let matmul_ty (et : Type0) {| scalar et |} : Type0 =
  (#rows : szp) ->
  (#shared : szp) ->
  (#cols : szp) ->
  (a : vec et) ->
  (b : vec et) ->
  (#sa : erased (seq et){ len sa == rows * shared }) ->
  (#sb : erased (seq et){ len sb == shared * cols }) ->
  stt (vec et)
  (requires
    (cpu **
    (a |-> sa) **
    (b |-> sb)) **
    (pure (SZ.fits (rows * shared) /\ SZ.fits (shared * cols) /\ SZ.fits (rows * cols)) **
    pure (rows * cols <= max_blocks)))
  (ensures fun c ->
    (cpu **
    (a |-> sa) **
    (b |-> sb)) **
    (c |-> to_row_major_seq <| MS.matmul (from_row_major_seq #_ #rows #shared sa) (from_row_major_seq #_ #shared #cols sb)))

inline_for_extraction noextract
val matmul
  (#et : Type0) {| scalar et |}
  (kk : kernel_ty et #_)
  : matmul_ty et
