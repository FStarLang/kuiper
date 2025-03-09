module Kuiper.MatMul

#lang-pulse

open Kuiper
module M = Kuiper.Matrix.Poly
module MS = Kuiper.Spec.MatMul
module SZ = FStar.SizeT
open Kuiper.EMatrix

inline_for_extraction
val kernel_fixed_ty
  (et : Type0) {| scalar et |}
  (rA rB rC : M.mrepr)
  (rows : nat)
  (shared : nat)
  (cols : nat{SZ.fits (rows * cols)})
  {| M.clayout (rA #rows #shared) |}
  {| M.clayout (rB #shared #cols) |}
  {| M.clayout (rC #rows #cols) |}
: Type0

inline_for_extraction
type kernel_ty
  (et : Type0) {| scalar et |}
  (rA rB rC : M.mrepr)
  {| cA : M.crepr rA |}
  {| cB : M.crepr rB |}
  {| cC : M.crepr rC |}
=
  (#rows:szp) ->
  (#shared:szp) ->
  (#cols:szp{SZ.fits (rows * cols) /\ SZ.fits (rows * shared) /\ SZ.fits (shared * cols)}) ->
  kernel_fixed_ty et rA rB rC rows shared cols #(cA.map rows shared) #_ #(cC.map rows cols)

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
