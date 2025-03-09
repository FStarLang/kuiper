module Kuiper.MatMul

#lang-pulse

open Kuiper
module M = Kuiper.Matrix.Poly
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT
open Kuiper.EMatrix

inline_for_extraction
val kernel_ty
  (et : Type0) {| scalar et |}
  (rA rB rC : M.mrepr)
  {| M.crepr rA |}
  {| M.crepr rB |}
  {| M.crepr rC |}
  : Type0

inline_for_extraction noextract
val kernel
  (#et : Type0) {| scalar et |}
  (rA rB rC : M.mrepr)
  {| M.crepr rA |}
  {| M.crepr rB |}
  {| M.crepr rC |}
  : kernel_ty et rA rB rC

unfold
let matmul_ty (et : Type0) {| scalar et |}
  (rA rB rC : M.mrepr)
  {| M.crepr rA |}
  {| M.crepr rB |}
  {| M.crepr rC |}
  : Type0
  =
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
    (c |-> M.to_seq rC <| MS.matmul (M.from_seq #_ #rows #shared rA sa)
                                    (M.from_seq #_ #shared #cols rB sb)))

inline_for_extraction noextract
val matmul
  (#et : Type0) {| scalar et |}
  (#rA #rB #rC : M.mrepr)
  {| M.crepr rA |}
  {| M.crepr rB |}
  {| M.crepr rC |}
  (kk : kernel_ty et rA rB rC)
  : matmul_ty et rA rB rC
