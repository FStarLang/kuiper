module Kuiper.MatMul

#lang-pulse

open Kuiper
module M = Kuiper.Matrix
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT

inline_for_extraction
val kernel_ty (et : Type0) {| scalar et |} : Type0

inline_for_extraction noextract
val kernel (#et : Type0) {| scalar et |} : kernel_ty et

unfold
let seq_as_matrix
  (#et : Type0)
  (rows cols : nat)
  (s : seq et{len s == rows * cols})
  : M.ematrix et rows cols
  = M.M <| s

unfold
let matrix_as_seq
  (#et : Type0)
  (#rows #cols : nat)
  (m : M.ematrix et rows cols)
  : GTot (s : seq et{len s == rows * cols})
  = m.s

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
    (c |-> matrix_as_seq <| MS.matmul (seq_as_matrix rows shared sa) (seq_as_matrix shared cols sb)))

inline_for_extraction noextract
val matmul
  (#et : Type0) {| scalar et |}
  (kk : kernel_ty et #_)
  : matmul_ty et
