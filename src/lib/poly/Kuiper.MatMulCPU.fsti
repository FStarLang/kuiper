module Kuiper.MatMulCPU

#lang-pulse

(* Invoking MatMulGPU, providing a wrapper callable from CPU code. *)

open Kuiper
open Kuiper.Matrix.Common
open Kuiper.MatMulGPU.Type
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
module MS = Kuiper.Spec.MatMul
module M  = Kuiper.Matrix

(* Fully polymorphic. No need to play tricks at this stage. *)
unfold
inline_for_extraction
type matmul_cpu_ty
=
  (#et : Type0) ->
  {| scalar et |} ->
  (#rows : szp) ->
  (#shared : szp) -> (* concrete args *)
  (#cols : szp{three_fits rows shared cols}) ->
  (#lA : mlayout rows shared) ->
  (#lB : mlayout shared cols) ->
  (#lC : mlayout rows cols) ->
  {| cA : clayout lA |} ->
  {| cB : clayout lB |} ->
  {| cC : clayout lC |} ->
  (a : vec et) ->
  (b : vec et) ->
  (#sa : erased (seq et){ len sa == rows * shared }) ->
  (#sb : erased (seq et){ len sb == shared * cols }) ->
  stt (vec et)
    (requires
      (cpu ** (a |-> sa) ** (b |-> sb)) **
      (* Would be better to parametrize this. The fact about rows * cols <= max_blocks
        is not needed for all kernels. *)
      (pure (three_fits rows shared cols) **
      pure (rows * cols <= max_blocks)))
    (ensures fun c ->
      (cpu ** (a |-> sa) ** (b |-> sb)) **
      (c |-> to_seq lC <|
                MS.matmul (from_seq lA sa)
                          (from_seq lB sb)))

inline_for_extraction noextract
val mk_matmul (matmul_gpu : matmul_gpu_ty) : matmul_cpu_ty

unfold
inline_for_extraction
type fixed_repr_matmul_cpu_ty
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA |}
  {| crepr rB |}
  {| crepr rC |}
=
  (#rows : szp) ->
  (#shared : szp) -> (* concrete args *)
  (#cols : szp{three_fits rows shared cols}) ->
  (a : vec et) ->
  (b : vec et) ->
  (#sa : erased (seq et){ len sa == rows * shared }) ->
  (#sb : erased (seq et){ len sb == shared * cols }) ->
  stt (vec et)
    (requires
      (cpu ** (a |-> sa) ** (b |-> sb)) **
      (* Would be better to parametrize this. The fact about rows * cols <= max_blocks
        is not needed for all kernels. *)
      (pure (three_fits rows shared cols) **
      pure (rows * cols <= max_blocks)))
    (ensures fun c ->
      (cpu ** (a |-> sa) ** (b |-> sb)) **
      (c |-> to_seq (rC rows cols) <|
                MS.matmul (from_seq (rA rows shared) sa)
                          (from_seq (rB shared cols) sb)))

inline_for_extraction noextract
val mk_fixed_repr_matmul
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| cA : crepr rA |}
  {| cB : crepr rB |}
  {| cC : crepr rC |}
  (matmul_gpu : (
    rows:szp ->
    shared:szp ->
    cols:szp{three_fits rows shared cols} ->
    matmul_gpu_ty_type_dims_repr et (rA rows shared) (rB shared cols) (rC rows cols) #(cA.map _ _) #(cB.map _ _) #(cC.map _ _)
  ))
  : fixed_repr_matmul_cpu_ty et rA rB rC
