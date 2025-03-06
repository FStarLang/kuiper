module Kuiper.MatMul

#lang-pulse

open Kuiper
module SZ = FStar.SizeT

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
  (#sa : erased (seq et)) ->
  (#sb : erased (seq et)) ->
  stt (vec et)
  (requires
    (cpu **
    (a |-> sa) **
    (b |-> sb)) **
    (pure (SZ.fits (rows * shared) /\ SZ.fits (shared * cols) /\ SZ.fits (rows * cols)) **
    pure (rows * cols <= max_blocks) **
    pure (len sa == rows * shared) **
    pure (len sb == shared * cols)))
  (ensures fun c ->
    (cpu **
    (a |-> sa) **
    (b |-> sb)) **
    (exists* sc. c |-> sc))

inline_for_extraction noextract
val matmul
  (#et : Type0) {| scalar et |}
  (kk : kernel_ty et #_)
  : matmul_ty et
