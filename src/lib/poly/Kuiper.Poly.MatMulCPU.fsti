module Kuiper.Poly.MatMulCPU

#lang-pulse

(* Invoking MatMulGPU, providing a wrapper callable from CPU code. *)

open Kuiper
open Kuiper.Matrix.Common
open Kuiper.Poly.MatMulGPU.Type
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
module MS = Kuiper.Spec.MatMul
open Kuiper.Matrix { gpu_matrix }

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
val matmul_cpu
  (matmul_gpu : matmul_gpu_ty)
  : matmul_cpu_ty

(* Does dynamic checks to ensure that the dimensions are multiples of tile. *)
inline_for_extraction noextract
val matmul_gpu_tiled
  (matmul_gpu : tiled_matmul_gpu_ty)
  (tile : valid_tile)
  : matmul_gpu_ty

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

unfold
inline_for_extraction
type fixed_repr_matmul_gpu_ty
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA |}
  {| crepr rB |}
  {| crepr rC |}
=
  (#rows : szp) ->
  (#shared : szp) -> (* concrete args *)
  (#cols : szp{three_fits rows shared cols}) ->
  (gA : gpu_matrix et (rA rows shared)) ->
  (gB : gpu_matrix et (rB shared cols)) ->
  (gC : gpu_matrix et (rC rows cols)) ->
  (#ma : ematrix et rows shared) ->
  (#mb : ematrix et shared cols) ->
  (#mc0 : ematrix et rows cols) ->
  stt unit
    (requires
      (cpu ** (gA |-> ma) ** (gB |-> mb)) **
      (* Would be better to parametrize this. The fact about rows * cols <= max_blocks
        is not needed for all kernels. *)
      (pure (three_fits rows shared cols) **
      pure (rows * cols <= max_blocks) **
      (gC |-> mc0)))
    (ensures fun _ ->
      (cpu ** (gA |-> ma) ** (gB |-> mb)) **
      (gC |-> MS.matmul ma mb))

inline_for_extraction noextract
val specialize_to_type_and_reprs_cpu
  (matmul_gpu : matmul_gpu_ty)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| cA : crepr rA |}
  {| cB : crepr rB |}
  {| cC : crepr rC |}
  : fixed_repr_matmul_cpu_ty et rA rB rC #cA #cB #cC

inline_for_extraction noextract
val specialize_to_type_and_reprs_gpu
  (matmul_gpu : matmul_gpu_ty)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| cA : crepr rA |}
  {| cB : crepr rB |}
  {| cC : crepr rC |}
  : fixed_repr_matmul_gpu_ty et rA rB rC #cA #cB #cC
