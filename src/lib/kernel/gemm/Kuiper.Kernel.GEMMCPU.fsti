module Kuiper.Kernel.GEMMCPU

#lang-pulse

(* Invoking GEMMGPU, providing a wrapper callable from CPU code. *)

open Kuiper
open Kuiper.Tensor
open Kuiper.Kernel.GEMMGPU.Type
open Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
include Kuiper.Kernel.GEMMGPU.Type { size_req_t }

(* Fully polymorphic. No need to play tricks at this stage. *)
unfold
inline_for_extraction
type matmul_cpu_ty
  (size_req : size_req_t)
=
  fn (#et : Type0) {| scalar et |}
    (#m #n #k : szp) (* concrete args *)
    (#lA : full_layout2 m k)
    (#lB : full_layout2 k n)
    (#lC : full_layout2 m n)
    {| ctlayout lA, ctlayout lB, ctlayout lC |}
    (a b : vec et)
    (#sa : erased (seq et){ len sa == m * k })
    (#sb : erased (seq et){ len sb == k * n })
  norewrite
  preserves
    cpu ** a |-> sa ** b |-> sb
  requires
    pure (size_req m n k)
  returns
    c : vec et
  ensures
    (* This will be nicer when views are usable on CPU arrays. *)
    c |-> (to_seq lC <|
             MS.matmul (from_seq lA sa)
                       (from_seq lB sb))

inline_for_extraction noextract
val matmul_cpu
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  : matmul_cpu_ty size_req

(* Does dynamic checks to ensure that the dimensions are multiples of tile.
This could maybe also be baked into the size_req to guarantee it statically. *)
inline_for_extraction noextract
val mmcomb_gpu_tiled
  (#size_req : tiled_size_req_t)
  (mmcomb_gpu : tiled_matmulcomb_gpu_ty size_req)
  (tile : valid_tile)
  : matmulcomb_gpu_ty
     (fun m n k ->
       size_req (m / tile) (n / tile) (k / tile) tile)

// (* Approximate version of mmcomb_gpu_tiled: wraps a tiled approximate GEMM
//    into a full-dimension approximate GEMM with external real matrices. *)
// inline_for_extraction noextract
// fn mmcomb_gpu_tiled_approx
//   (#size_req : tiled_size_req_t)
//   (mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req)
//   (tile : valid_tile)
//   (#et : Type0) {| scalar et |} {| real_like et |}
//   (comb : binop et)
//   (comb_r : binop real { approx2 comb comb_r })
//   (#m #n #k : szp)
//   (#lA : full_mlayout m k)
//   (#lB : full_mlayout k n)
//   (#lC : full_mlayout m n)
//   {| clayout lA, clayout lB, clayout lC |}
//   (gA : gpu_matrix et lA { is_global gA })
//   (#fA : perm)
//   (gB : gpu_matrix et lB { is_global gB })
//   (#fB : perm)
//   (gC : gpu_matrix et lC { is_global gC })
//   (#eA : chest2 et m k)
//   (#eB : chest2 et k n)
//   (#eC : chest2 et m n)
//   (rA : chest2 real m k)
//   (rB : chest2 real k n)
//   (rC : chest2 real m n)
//   requires
//     (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
//     (pure (size_req (m / tile) (k / tile) (n / tile) tile) **
//      pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
//      on gpu_loc (gC |-> eC))
//   ensures
//     (cpu ** on gpu_loc (gA |-> Frac fA eA) ** on gpu_loc (gB |-> Frac fB eB)) **
//     (exists* (eC' : chest2 et m n).
//       on gpu_loc (gC |-> eC') **
//       pure (eC' %~ MS.mmcomb comb_r rC rA rB))

// unfold
// inline_for_extraction
// type fixed_repr_matmul_cpu_ty
//   (et : Type0) {| scalar et |}
//   (size_req : size_req_t)
//   (rA rB rC : mrepr)
//   {| crepr rA, crepr rB, crepr rC |}
// =
//   fn (#m : szp)
//      (#k : szp) (* concrete args *)
//      (#n : szp)
//      (a : vec et)
//      (b : vec et)
//      (#sa : erased (seq et){ len sa == m * k })
//      (#sb : erased (seq et){ len sb == k * n })
//   norewrite
//   preserves
//     cpu **
//     a |-> sa **
//     b |-> sb
//   requires
//     pure (size_req m n k) **
//     pure (SZ.fits (m * n))
//   returns
//     c : vec et
//   ensures
//     c |-> (to_seq (rC m n) <|
//               MS.matmul (from_seq (rA m k) sa)
//                         (from_seq (rB k n) sb))

// unfold
// inline_for_extraction
// type fixed_repr_gemm_gpu_ty
//   (et : Type0) {| scalar et |}
//   (size_req : size_req_t)
//   (rA rB rC : mrepr)
//   {| crepr rA, crepr rB, crepr rC |}
// =
//   fn (alpha : et)
//      (beta : et)
//      (#m : szp)
//      (#k : szp) (* concrete args *)
//      (#n : szp)
//      (gA : gpu_matrix et (rA m k) { is_global gA})
//      (gB : gpu_matrix et (rB k n) { is_global gB})
//      (gC : gpu_matrix et (rC m n) { is_global gC})
//      (#ma : chest2 et m k)
//      (#mb : chest2 et k n)
//      (#mc0 : chest2 et m n)
//   norewrite
//   preserves
//     cpu **
//     on gpu_loc (gA |-> ma) **
//     on gpu_loc (gB |-> mb)
//   requires
//     pure (size_req m n k) **
//     on gpu_loc (gC |-> mc0)
//   ensures
//     on gpu_loc (gC |-> MS.gemm alpha beta mc0 ma mb)

// (* Approximate GEMM GPU type: postcondition approximates real-valued gemm. *)
// unfold
// inline_for_extraction
// type fixed_repr_gemm_gpu_approx_ty
//   (et : Type0) {| scalar et, real_like et |}
//   (size_req : size_req_t)
//   (repA repB repC : mrepr)
//   {| crepr repA, crepr repB, crepr repC |}
// =
//   fn (alpha : et)
//      (beta : et)
//      (alpha_r : real)
//      (beta_r : real)
//      (#m : szp)
//      (#k : szp)
//      (#n : szp)
//      (gA : gpu_matrix et (repA m k) { is_global gA })
//      (gB : gpu_matrix et (repB k n) { is_global gB })
//      (gC : gpu_matrix et (repC m n) { is_global gC })
//      (#ma : chest2 et m k)
//      (#mb : chest2 et k n)
//      (#mc0 : chest2 et m n)
//      (rA : chest2 real m k)
//      (rB : chest2 real k n)
//      (rC : chest2 real m n)
//   norewrite
//   preserves
//     cpu **
//     on gpu_loc (gA |-> ma) **
//     on gpu_loc (gB |-> mb)
//   requires
//     pure (size_req m n k) **
//     pure (ma %~ rA /\ mb %~ rB /\ mc0 %~ rC /\
//           alpha %~ alpha_r /\ beta %~ beta_r) **
//     on gpu_loc (gC |-> mc0)
//   ensures (
//     exists* (mc' : chest2 et m n).
//       on gpu_loc (gC |-> mc') **
//       pure (mc' %~ MS.gemm (alpha_r) (beta_r) rC rA rB))

// unfold
// inline_for_extraction
// type fixed_repr_mmcomb_gpu_ty
//   (et : Type0) {| scalar et |}
//   (size_req : size_req_t)
//   (rA rB rC : mrepr)
//   {| crepr rA, crepr rB, crepr rC |}
// =
//   fn (#m : szp)
//      (#k : szp) (* concrete args *)
//      (#n : szp)
//      (gA : gpu_matrix et (rA m k) { is_global gA })
//      (gB : gpu_matrix et (rB k n) { is_global gB })
//      (gC : gpu_matrix et (rC m n) { is_global gC })
//      (#ma : chest2 et m k)
//      (#mb : chest2 et k n)
//      (#mc0 : chest2 et m n)
//   norewrite
//   preserves
//     cpu **
//     on gpu_loc (gA |-> ma) **
//     on gpu_loc (gB |-> mb)
//   requires
//     pure (size_req m n k) **
//     on gpu_loc (gC |-> mc0)
//   ensures
//     on gpu_loc (gC |-> MS.matmul ma mb)

// inline_for_extraction noextract
// val specialize_as_gemm_to_type_and_reprs_gpu
//   (#size_req : size_req_t)
//   (mmcomb_gpu : matmulcomb_gpu_ty size_req)
//   (et : Type0) {| scalar et |}
//   (rA rB rC : mrepr)
//   {| crepr rA, crepr rB, crepr rC |}
//   : fixed_repr_gemm_gpu_ty et size_req rA rB rC

// inline_for_extraction noextract
// val specialize_as_matmul_to_type_and_reprs_gpu
//   (#size_req : size_req_t)
//   (mmcomb_gpu : matmulcomb_gpu_ty size_req)
//   (et : Type0) {| scalar et |}
//   (rA rB rC : mrepr)
//   {| crepr rA, crepr rB, crepr rC |}
//   : fixed_repr_mmcomb_gpu_ty et size_req rA rB rC

// (* Approximate GPU matmul type: postcondition is eC' %~ MS.matmul rA rB
//    where rA, rB are real-valued matrices. *)
// unfold
// inline_for_extraction
// type fixed_repr_mmcomb_gpu_approx_ty
//   (et : Type0) {| scalar et, real_like et |}
//   (size_req : size_req_t)
//   (repA repB repC : mrepr)
//   {| crepr repA, crepr repB, crepr repC |}
// =
//   fn (#m : szp)
//      (#k : szp)
//      (#n : szp)
//      (gA : gpu_matrix et (repA m k) { is_global gA })
//      (gB : gpu_matrix et (repB k n) { is_global gB })
//      (gC : gpu_matrix et (repC m n) { is_global gC })
//      (#ma : chest2 et m k)
//      (#mb : chest2 et k n)
//      (#mc0 : chest2 et m n)
//      (rA : chest2 real m k)
//      (rB : chest2 real k n)
//   norewrite
//   preserves
//     cpu **
//     on gpu_loc (gA |-> ma) **
//     on gpu_loc (gB |-> mb)
//   requires
//     pure (size_req m n k) **
//     pure (ma %~ rA /\ mb %~ rB) **
//     on gpu_loc (gC |-> mc0)
//   ensures (
//     exists* (mc' : chest2 et m n).
//       on gpu_loc (gC |-> mc') **
//       pure (mc' %~ MS.matmul rA rB))

// inline_for_extraction noextract
// val specialize_tiled_approx_gpu
//   (#size_req : tiled_size_req_t)
//   (mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req)
//   (tile : valid_tile)
//   (et : Type0) {| scalar et |} {| real_like et |}
//   (repA repB repC : mrepr)
//   {| crepr repA, crepr repB, crepr repC |}
//   : fixed_repr_mmcomb_gpu_approx_ty et
//       (fun m n k ->
//         size_req (m / tile) (k / tile) (n / tile) tile)
//       repA repB repC

// inline_for_extraction noextract
// val specialize_tiled_approx_gemm_gpu
//   (#size_req : tiled_size_req_t)
//   (mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req)
//   (tile : valid_tile)
//   (et : Type0) {| scalar et |} {| real_like et |}
//   (repA repB repC : mrepr)
//   {| crepr repA, crepr repB, crepr repC |}
//   : fixed_repr_gemm_gpu_approx_ty et
//       (fun m n k ->
//         size_req (m / tile) (k / tile) (n / tile) tile)
//       repA repB repC

// inline_for_extraction noextract
// val specialize_as_matmul_to_type_and_reprs_cpu
//   (#size_req : size_req_t)
//   (mmcomb_gpu : matmulcomb_gpu_ty size_req)
//   (et : Type0) {| scalar et |}
//   (rA rB rC : mrepr)
//   {| crepr rA, crepr rB, crepr rC |}
//   : fixed_repr_matmul_cpu_ty et size_req rA rB rC

// (* Approximate CPU matmul type: result vec approximates the real-valued matmul. *)
// unfold
// inline_for_extraction
// type fixed_repr_matmul_cpu_approx_ty
//   (et : Type0) {| scalar et, real_like et |}
//   (size_req : size_req_t)
//   (repA repB repC : mrepr)
//   {| crepr repA, crepr repB, crepr repC |}
// =
//   fn (#m : szp)
//      (#k : szp)
//      (#n : szp)
//      (a : vec et)
//      (b : vec et)
//      (#sa : erased (seq et){ len sa == m * k })
//      (#sb : erased (seq et){ len sb == k * n })
//      (rA : chest2 real m k)
//      (rB : chest2 real k n)
//   norewrite
//   preserves
//     cpu **
//     a |-> sa **
//     b |-> sb
//   requires
//     pure (size_req m n k) **
//     pure (SZ.fits (m * n)) **
//     pure (from_seq (repA m k) sa %~ rA /\
//           from_seq (repB k n) sb %~ rB)
//   returns
//     c : vec et
//   ensures (
//     exists* (sc : seq et).
//       c |-> sc **
//       pure (len sc == m * n /\
//             from_seq (repC m n) sc %~ MS.matmul rA rB))

// inline_for_extraction noextract
// val specialize_tiled_approx_cpu
//   (#size_req : tiled_size_req_t)
//   (mmcomb_gpu_approx : tiled_matmulcomb_gpu_approx_ty size_req)
//   (tile : valid_tile)
//   (et : Type0) {| scalar et, real_like et |}
//   (repA repB repC : mrepr)
//   {| crepr repA, crepr repB, crepr repC |}
//   : fixed_repr_matmul_cpu_approx_ty et
//       (fun m n k -> size_req (m / tile) (n / tile) (k / tile) tile)
//       repA repB repC
