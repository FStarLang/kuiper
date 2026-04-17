module Kuiper.GEMM.BlockTiling1D

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.Poly.GEMMGPU.Type
module M = Kuiper.Array2
open Kuiper.Tensor.Layout.Alg
open Kuiper.EMatrix
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module P = Kuiper.Poly.GEMM.BlockTiling1D

// Note: all row major until we support layout families better
inline_for_extraction noextract
fn spec
  (tile : valid_tile)
  (et : Type0) {| scalar et, real_like et |}
  (comb : binop et) (comb_r : binop real { comb `approx2` comb_r })
  (m n k : szp)
  (gA : M.array2 et (l2_row_major m k) { M.is_global gA })
  (gB : M.array2 et (l2_row_major k n) { M.is_global gB })
  (gC : M.array2 et (l2_row_major m n) { M.is_global gC })
  (rA rB rC : ematrix real _ _)
  (#eA #eB : ematrix _ _ _)
  (#eC : ematrix et m n)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB) **
    pure (tile /? m /\ tile /? n /\ tile /? k)
  requires
    pure (P.size_req m n k tile) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures (
    (exists* (eC' : ematrix et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB)))
{
  map_loc gpu_loc (fun () -> M.pts_to_ref gA);
  map_loc gpu_loc (fun () -> M.pts_to_ref gB);
  map_loc gpu_loc (fun () -> M.pts_to_ref gC);

  let mm = m /^ tile;
  let nn = n /^ tile;
  let kk = k /^ tile;

  P.mmcomb_gpu_approx tile
    comb comb_r
    #mm #nn #kk
    #(l2_row_major m k) #(l2_row_major k n) #(l2_row_major m n)
    #(c_l2_row_major (SZ.v m) k)
    #(c_l2_row_major (SZ.v k) n)
    #(c_l2_row_major (SZ.v m) n)
    gA gB gC
    rA rB rC;

  ()
}

inline_for_extraction noextract
let spec_mm (tile : valid_tile) (et : Type0) {| scalar et, real_like et |} =
  spec tile et (fun _o n -> n) (fun _o n -> n)

inline_for_extraction noextract
let spec_gemm (tile : valid_tile) (et : Type0) {| scalar et, real_like et |}
  (alpha beta : et) =
  to_real_ok alpha;
  to_real_ok beta;
  spec tile et
    (MS.lincomb alpha beta) (MS.lincomb (to_real alpha) (to_real beta))

(* No dynamic tile size here: the algorithms uses a local
array (registers) of size tile, and CUDA requires these sizes
to be statically known. *)

(* dynamic tile *)
// let g_matmul_f32_rrr tile = spec_mm tile f32
// let g_matmul_f64_rrr tile = spec_mm tile f64
// let g_matmul_u32_rrr tile = spec_mm tile u32
// let g_matmul_u64_rrr tile = spec_mm tile u64

(* tile=32 *)
let g_matmul_f32_tile32_rrr = spec_mm 32sz f32
let g_matmul_f64_tile32_rrr = spec_mm 32sz f64
let g_matmul_u32_tile32_rrr = spec_mm 32sz u32
let g_matmul_u64_tile32_rrr = spec_mm 32sz u64

(* tile=16 *)
let g_matmul_f32_tile16_rrr = spec_mm 16sz f32
let g_matmul_f64_tile16_rrr = spec_mm 16sz f64
let g_matmul_u32_tile16_rrr = spec_mm 16sz u32
let g_matmul_u64_tile16_rrr = spec_mm 16sz u64

(* dynamic tile *)
// let g_gemm_f32_rrr tile = spec_gemm tile f32
// let g_gemm_f64_rrr tile = spec_gemm tile f64
// let g_gemm_u32_rrr tile = spec_gemm tile u32
// let g_gemm_u64_rrr tile = spec_gemm tile u64

(* tile=32 *)
let g_gemm_f32_tile32_rrr = spec_gemm 32sz f32
let g_gemm_f64_tile32_rrr = spec_gemm 32sz f64
let g_gemm_u32_tile32_rrr = spec_gemm 32sz u32
let g_gemm_u64_tile32_rrr = spec_gemm 32sz u64

(* tile=16 *)
let g_gemm_f32_tile16_rrr = spec_gemm 16sz f32
let g_gemm_f64_tile16_rrr = spec_gemm 16sz f64
let g_gemm_u32_tile16_rrr = spec_gemm 16sz u32
let g_gemm_u64_tile16_rrr = spec_gemm 16sz u64
