module Kuiper.MatMul

#lang-pulse

open Kuiper
module M = Kuiper.Matrix
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

inline_for_extraction
val kernel_fixed_ty
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  (rows : nat)
  (shared : nat)
  (cols : nat{SZ.fits (rows * cols)})
: Type0

inline_for_extraction
type kernel_ty
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| cA : crepr rA |}
  {| cB : crepr rB |}
  {| cC : crepr rC |}
=
  (#rows:szp) ->
  (#shared:szp) ->
  (#cols:szp{SZ.fits (rows * cols) /\ SZ.fits (rows * shared) /\ SZ.fits (shared * cols)}) ->
  kernel_fixed_ty et rA rB rC rows shared cols

inline_for_extraction noextract
val kernel
  (#et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA |}
  {| crepr rB |}
  {| crepr rC |}
  : kernel_ty et rA rB rC

unfold
let matmul_ty (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA |}
  {| crepr rB |}
  {| crepr rC |}
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
    (c |-> M.to_seq (rC rows cols) <|
             MS.matmul (M.from_seq (rA rows shared) sa)
                       (M.from_seq (rB shared cols) sb)))

inline_for_extraction noextract
val matmul
  (#et : Type0) {| scalar et |}
  (#rA #rB #rC : mrepr)
  {| crepr rA |}
  {| crepr rB |}
  {| crepr rC |}
  (kk : kernel_ty et rA rB rC)
  : matmul_ty et rA rB rC
