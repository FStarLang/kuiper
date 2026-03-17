module Kuiper.Poly.GEMMCPU

#lang-pulse

(* Invoking GEMMGPU, providing a wrapper callable from CPU code. *)

open Kuiper
open Kuiper.Approximates
open Kuiper.Matrix.Common
open Kuiper.Poly.GEMMGPU.Type
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
open Kuiper.Matrix
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

(* Approximate version of mmcomb_gpu_tiled: wraps a tiled approximate GEMM
   into a full-dimension approximate GEMM with external real matrices. *)
inline_for_extraction noextract
val mmcomb_gpu_tiled_approx
  (#size_req : tiled_size_req_t)
  (mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req)
  (tile : valid_tile)
  (#et : Type0) {| scalar et |} {| real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#rows #shared #cols : szp)
  (#lA : full_mlayout rows shared)
  (#lB : full_mlayout shared cols)
  (#lC : full_mlayout rows cols)
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  : stt unit
    (requires
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (pure (size_req (rows / tile) (shared / tile) (cols / tile) tile) **
       pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
       on gpu_loc (gC |-> eC)))
    (ensures fun _ ->
      (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
      (exists* (eC' : ematrix et rows cols).
        on gpu_loc (gC |-> eC') **
        pure (eC' %~ MS.mmcomb comb_r rC rA rB)))

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
  (gA : gpu_matrix et (rA rows shared) { is_global_matrix gA}) ->
  (gB : gpu_matrix et (rB shared cols) { is_global_matrix gB}) ->
  (gC : gpu_matrix et (rC rows cols) { is_global_matrix gC}) ->
  (#ma : ematrix et rows shared) ->
  (#mb : ematrix et shared cols) ->
  (#mc0 : ematrix et rows cols) ->
  stt unit
    (requires
      (cpu ** on gpu_loc (gA |-> ma) ** on gpu_loc (gB |-> mb)) **
      (pure (size_req rows shared cols) **
       on gpu_loc (gC |-> mc0)))
    (ensures fun _ ->
      (cpu ** on gpu_loc (gA |-> ma) ** on gpu_loc (gB |-> mb)) **
      (on gpu_loc (gC |-> MS.gemm alpha beta mc0 ma mb)))

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
  (gA : gpu_matrix et (rA rows shared) { is_global_matrix gA }) ->
  (gB : gpu_matrix et (rB shared cols) { is_global_matrix gB }) ->
  (gC : gpu_matrix et (rC rows cols) { is_global_matrix gC }) ->
  (#ma : ematrix et rows shared) ->
  (#mb : ematrix et shared cols) ->
  (#mc0 : ematrix et rows cols) ->
  stt unit
    (requires
      (cpu ** on gpu_loc (gA |-> ma) ** on gpu_loc (gB |-> mb)) **
      (pure (size_req rows shared cols) **
       on gpu_loc (gC |-> mc0)))
    (ensures fun _ ->
      (cpu ** on gpu_loc (gA |-> ma) ** on gpu_loc (gB |-> mb)) **
      (on gpu_loc (gC |-> MS.matmul ma mb)))

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

(* Approximate GPU matmul type: postcondition is eC' %~ MS.matmul rA rB
   where rA, rB are real-valued matrices. *)
unfold
inline_for_extraction
type fixed_repr_mmcomb_gpu_approx_ty
  (et : Type0) {| scalar et, real_like et |}
  (size_req : size_req_t)
  (repA repB repC : mrepr)
  {| crepr repA, crepr repB, crepr repC |}
=
  (#rows : szp) ->
  (#shared : szp) ->
  (#cols : szp) ->
  (gA : gpu_matrix et (repA rows shared) { is_global_matrix gA }) ->
  (gB : gpu_matrix et (repB shared cols) { is_global_matrix gB }) ->
  (gC : gpu_matrix et (repC rows cols) { is_global_matrix gC }) ->
  (#ma : ematrix et rows shared) ->
  (#mb : ematrix et shared cols) ->
  (#mc0 : ematrix et rows cols) ->
  (rA : ematrix real rows shared) ->
  (rB : ematrix real shared cols) ->
  stt unit
    (requires
      (cpu ** on gpu_loc (gA |-> ma) ** on gpu_loc (gB |-> mb)) **
      (pure (size_req rows shared cols) **
       pure (ma %~ rA /\ mb %~ rB) **
       on gpu_loc (gC |-> mc0)))
    (ensures fun _ ->
      (cpu ** on gpu_loc (gA |-> ma) ** on gpu_loc (gB |-> mb)) **
      (exists* (mc' : ematrix et rows cols).
        on gpu_loc (gC |-> mc') **
        pure (mc' %~ MS.matmul rA rB)))

inline_for_extraction noextract
val specialize_tiled_approx_gpu
  (#size_req : tiled_size_req_t)
  (mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req)
  (tile : valid_tile)
  (et : Type0) {| scalar et |} {| real_like et |}
  (repA repB repC : mrepr)
  {| crepr repA, crepr repB, crepr repC |}
  : fixed_repr_mmcomb_gpu_approx_ty et
      (fun rows shared cols ->
        size_req (rows / tile) (shared / tile) (cols / tile) tile)
      repA repB repC

inline_for_extraction noextract
val specialize_as_matmul_to_type_and_reprs_cpu
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (et : Type0) {| scalar et |}
  (rA rB rC : mrepr)
  {| crepr rA, crepr rB, crepr rC |}
  : fixed_repr_matmul_cpu_ty et size_req rA rB rC

(* Approximate CPU matmul type: result vec approximates the real-valued matmul. *)
unfold
inline_for_extraction
type fixed_repr_matmul_cpu_approx_ty
  (et : Type0) {| scalar et, real_like et |}
  (size_req : size_req_t)
  (repA repB repC : mrepr)
  {| crepr repA, crepr repB, crepr repC |}
=
  (#rows : szp) ->
  (#shared : szp) ->
  (#cols : szp) ->
  (a : vec et) ->
  (b : vec et) ->
  (#sa : erased (seq et){ len sa == rows * shared }) ->
  (#sb : erased (seq et){ len sb == shared * cols }) ->
  (rA : ematrix real rows shared) ->
  (rB : ematrix real shared cols) ->
  stt (vec et)
    (requires
      (cpu ** a |-> sa ** b |-> sb) **
       pure (size_req rows shared cols) **
       pure (SZ.fits (rows * cols)) **
       pure (from_seq (repA rows shared) sa %~ rA /\
             from_seq (repB shared cols) sb %~ rB))
    (ensures fun c ->
      (cpu ** a |-> sa ** b |-> sb) **
      (exists* (sc : seq et).
        c |-> sc **
        pure (len sc == rows * cols /\
              from_seq (repC rows cols) sc %~ MS.matmul rA rB)))

inline_for_extraction noextract
val specialize_tiled_approx_cpu
  (#size_req : tiled_size_req_t)
  (mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req)
  (tile : valid_tile)
  (et : Type0) {| scalar et |} {| real_like et |}
  (repA repB repC : mrepr)
  {| crepr repA, crepr repB, crepr repC |}
  : fixed_repr_matmul_cpu_approx_ty et
      (fun rows shared cols ->
        size_req (rows / tile) (shared / tile) (cols / tile) tile)
      repA repB repC
