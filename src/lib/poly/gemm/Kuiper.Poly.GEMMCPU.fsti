module Kuiper.Poly.GEMMCPU

#lang-pulse

(* Invoking GEMMGPU, providing a wrapper callable from CPU code. *)

open Kuiper
open Kuiper.Matrix.Common
open Kuiper.Poly.GEMMGPU.Type
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
module MS = Kuiper.Spec.GEMM
module SZ = FStar.SizeT
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
  (#cols : szp) ->
  (#lA : full_mlayout rows shared) ->
  (#lB : full_mlayout shared cols) ->
  (#lC : full_mlayout rows cols) ->
  {| cA : clayout lA |} ->
  {| cB : clayout lB |} ->
  {| cC : clayout lC |} ->
  (a : vec et) ->
  (b : vec et) ->
  (#sa : erased (seq et){ len sa == rows * shared }) ->
  (#sb : erased (seq et){ len sb == shared * cols }) ->
  stt (vec et)
    (requires
      (cpu ** a |-> sa ** b |-> sb) **
      (* Would be better to parametrize this. The fact about rows * cols <= max_blocks
        is not needed for all kernels. *)
      (pure (SZ.fits (rows * cols)) **
      pure (rows * cols <= max_blocks)))
    (ensures fun c ->
      (cpu ** a |-> sa ** b |-> sb) **
      (c |-> (to_seq lC <|
                MS.matmul (from_seq lA sa)
                          (from_seq lB sb))))

inline_for_extraction noextract
val matmul_cpu
  (mmcomb_gpu : matmulcomb_gpu_ty)
  : matmul_cpu_ty

(* Does dynamic checks to ensure that the dimensions are multiples of tile. *)
inline_for_extraction noextract
val mmcomb_gpu_tiled
  (mmcomb_gpu : tiled_matmulcomb_gpu_ty)
  (tile : valid_tile)
  : matmulcomb_gpu_ty

inline_for_extraction noextract
val mmcomb_gpu_block_tiled1d
  (mmcomb_gpu : block_tiled1d_matmulcomb_gpu_ty)
  (bm bn bk : szp)
  (tm : szp{tm /? bm /\ (bm/tm * bn < max_threads)})
  : matmulcomb_gpu_ty

inline_for_extraction noextract
val mmcomb_gpu_shmem_block_tiled2d
  (mmcomb_gpu : block_tiled2d_matmulcomb_gpu_ty)
  (bm bn bk : szp)
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| csA : clayout slA |}
  {| csB : clayout slB |}
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn /\ (bm/tm * bn/tn < max_threads)})
  : matmulcomb_gpu_ty

unfold
inline_for_extraction
type fixed_repr_matmul_cpu_ty
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA, crepr rB, crepr rC |}
=
  (#rows : szp) ->
  (#shared : szp) -> (* concrete args *)
  (#cols : szp) ->
  (a : vec et) ->
  (b : vec et) ->
  (#sa : erased (seq et){ len sa == rows * shared }) ->
  (#sb : erased (seq et){ len sb == shared * cols }) ->
  stt (vec et)
    (requires
      (cpu ** a |-> sa ** b |-> sb) **
      (* Would be better to parametrize this. The fact about rows * cols <= max_blocks
        is not needed for all kernels. *)
      (pure (SZ.fits (rows * cols)) **
      pure (rows * cols <= max_blocks)))
    (ensures fun c ->
      (cpu ** a |-> sa ** b |-> sb) **
      (c |-> (to_seq (rC rows cols) <|
                MS.matmul (from_seq (rA rows shared) sa)
                          (from_seq (rB shared cols) sb))))

unfold
inline_for_extraction
type fixed_repr_gemm_gpu_ty
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA, crepr rB, crepr rC |}
=
  (alpha : et) ->
  (beta : et) ->
  (#rows : szp) ->
  (#shared : szp) -> (* concrete args *)
  (#cols : szp) ->
  (gA : gpu_matrix et (rA rows shared)) ->
  (gB : gpu_matrix et (rB shared cols)) ->
  (gC : gpu_matrix et (rC rows cols)) ->
  (#ma : ematrix et rows shared) ->
  (#mb : ematrix et shared cols) ->
  (#mc0 : ematrix et rows cols) ->
  stt unit
    (requires
      (cpu ** gA |-> ma ** gB |-> mb) **
      (* Would be better to parametrize this. The fact about rows * cols <= max_blocks
        is not needed for all kernels. *)
      (pure (SZ.fits (rows * cols)) **
      pure (rows * cols <= max_blocks) **
      gC |-> mc0))
    (ensures fun _ ->
      (cpu ** gA |-> ma ** gB |-> mb) **
      (gC |-> MS.gemm alpha beta mc0 ma mb))

unfold
inline_for_extraction
type fixed_repr_mmcomb_gpu_ty
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA, crepr rB, crepr rC |}
=
  (#rows : szp) ->
  (#shared : szp) -> (* concrete args *)
  (#cols : szp) ->
  (gA : gpu_matrix et (rA rows shared)) ->
  (gB : gpu_matrix et (rB shared cols)) ->
  (gC : gpu_matrix et (rC rows cols)) ->
  (#ma : ematrix et rows shared) ->
  (#mb : ematrix et shared cols) ->
  (#mc0 : ematrix et rows cols) ->
  stt unit
    (requires
      (cpu ** gA |-> ma ** gB |-> mb) **
      (* Would be better to parametrize this. The fact about rows * cols <= max_blocks
        is not needed for all kernels. *)
      (pure (SZ.fits (rows * cols)) **
      pure (rows * cols <= max_blocks) **
      gC |-> mc0))
    (ensures fun _ ->
      (cpu ** gA |-> ma ** gB |-> mb) **
      (gC |-> MS.matmul ma mb))

inline_for_extraction noextract
val specialize_as_gemm_to_type_and_reprs_gpu
  (mmcomb_gpu : matmulcomb_gpu_ty)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA, crepr rB, crepr rC |}
  : fixed_repr_gemm_gpu_ty et rA rB rC

inline_for_extraction noextract
val specialize_as_matmul_to_type_and_reprs_gpu
  (mmcomb_gpu : matmulcomb_gpu_ty)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA, crepr rB, crepr rC |}
  : fixed_repr_mmcomb_gpu_ty et rA rB rC

// inline_for_extraction noextract
// val specialize_as_gemm_to_type_and_reprs_gpu
//   (mmcomb_gpu : mmcomb_gpu_ty)
//   (et : Type0) {| scalar et |}
//   (rA rB rC : mrepr)
//   {| cA : crepr rA |}
//   {| cB : crepr rB |}
//   {| cC : crepr rC |}
//   : fixed_repr_gemm_gpu_ty et rA rB rC #cA #cB #cC

inline_for_extraction noextract
val specialize_as_matmul_to_type_and_reprs_cpu
  (mmcomb_gpu : matmulcomb_gpu_ty)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA, crepr rB, crepr rC |}
  : fixed_repr_matmul_cpu_ty et rA rB rC
