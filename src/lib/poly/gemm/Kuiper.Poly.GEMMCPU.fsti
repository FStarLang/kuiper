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
include Kuiper.Poly.GEMMGPU.Type { size_req_t }

(* Fully polymorphic. No need to play tricks at this stage. *)
unfold
inline_for_extraction
type matmul_cpu_ty
  (size_req : size_req_t)
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
      pure (size_req rows shared cols))
    (ensures fun c ->
      (cpu ** a |-> sa ** b |-> sb) **
      c |-> (to_seq lC <|
                MS.matmul (from_seq lA sa)
                          (from_seq lB sb)))

inline_for_extraction noextract
val matmul_cpu
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  : matmul_cpu_ty size_req

(* Does dynamic checks to ensure that the dimensions are multiples of tile. *)
inline_for_extraction noextract
val mmcomb_gpu_tiled
  (#size_req : tiled_size_req_t)
  (mmcomb_gpu : tiled_matmulcomb_gpu_ty size_req)
  (tile : valid_tile)
  : matmulcomb_gpu_ty
     (fun rows shared cols ->
       size_req (rows / tile) (shared / tile) (cols / tile) tile)

inline_for_extraction noextract
val mmcomb_gpu_block_tiled1d
  (mmcomb_gpu : block_tiled1d_matmulcomb_gpu_ty)
  (bm bn bk : szp)
  (tm : szp{tm /? bm /\ (bm/tm * bn <= max_threads)})
  : matmulcomb_gpu_ty
     (fun rows _ cols -> (rows / bm) * (cols / bn) <= max_blocks)

// inline_for_extraction noextract
// fn mmcomb_gpu_shmem_block_tc
//   (tiled_mmcomb_gpu : block_tiled_tc_matmulcomb_gpu_ty)
//   (bm bn bk : szp)
//   (tm : szp{tm /? bm})
//   (tn : szp{tn /? bn /\ (bm/tm * bn/tn <= max_threads)})
//   (tk : szp{tk /? bk})
//   (#rows #shared #cols : szp)
//   (#lA : full_mlayout rows shared)
//   (#lB : full_mlayout shared cols)
//   {| cA : clayout lA |}
//   {| cB : clayout lB |}
//   (gA : M.gpu_matrix half lA)
//   (#fA : perm)
//   (gB : M.gpu_matrix half lB)
//   (#fB : perm)
//   (gC : M.gpu_matrix half (Kuiper.Matrix.Reprs.row_major rows cols))
//   (#eA : ematrix half rows shared)
//   (#eB : ematrix half shared cols)
//   (#eC : ematrix half rows cols)
//   norewrite
//   preserves
//     cpu **
//     gA |-> Frac fA eA **
//     gB |-> Frac fB eB
//   requires
//     pure (rows * cols <= max_blocks) **
//     gC |-> eC
//   ensures

unfold
inline_for_extraction
type fixed_repr_matmul_cpu_ty
  (et : Type0) {| scalar et |}
  (size_req : size_req_t)
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
       pure (size_req rows shared cols) **
       pure (SZ.fits (rows * cols)))
    (ensures fun c ->
      (cpu ** a |-> sa ** b |-> sb) **
      (c |-> (to_seq (rC rows cols) <|
                MS.matmul (from_seq (rA rows shared) sa)
                          (from_seq (rB shared cols) sb))))

unfold
inline_for_extraction
type fixed_repr_gemm_gpu_ty
  (et : Type0) {| scalar et |}
  (size_req : size_req_t)
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
      (pure (size_req rows shared cols) **
       gC |-> mc0))
    (ensures fun _ ->
      (cpu ** gA |-> ma ** gB |-> mb) **
      (gC |-> MS.gemm alpha beta mc0 ma mb))

unfold
inline_for_extraction
type fixed_repr_mmcomb_gpu_ty
  (et : Type0) {| scalar et |}
  (size_req : size_req_t)
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
      (pure (size_req rows shared cols) **
       gC |-> mc0))
    (ensures fun _ ->
      (cpu ** gA |-> ma ** gB |-> mb) **
      (gC |-> MS.matmul ma mb))

inline_for_extraction noextract
val specialize_as_gemm_to_type_and_reprs_gpu
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA, crepr rB, crepr rC |}
  : fixed_repr_gemm_gpu_ty et size_req rA rB rC

inline_for_extraction noextract
val specialize_as_matmul_to_type_and_reprs_gpu
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA, crepr rB, crepr rC |}
  : fixed_repr_mmcomb_gpu_ty et size_req rA rB rC

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
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA, crepr rB, crepr rC |}
  : fixed_repr_matmul_cpu_ty et size_req rA rB rC
