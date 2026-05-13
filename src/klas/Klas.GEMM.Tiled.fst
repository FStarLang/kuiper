module Klas.GEMM.Tiled

#lang-pulse
open Kuiper
open Kuiper.Kernel.GEMMGPU.Type { valid_tile }
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.EMatrix
open Kuiper.Array2
module MS = Kuiper.Spec.GEMM
module K = Kuiper.Kernel.GEMM.Tiled

inline_for_extraction noextract
fn spec
  (tile : valid_tile)
  (et : Type0) {| scalar et, real_like et |}
  (comb : binop et) (comb_r : binop real { comb `approx2` comb_r })
  (repA repB repC : trepr2)
  {| crepA : ctrepr2 repA, crepB : ctrepr2 repB, crepC : ctrepr2 repC |}
  (m n k : szp)
  (gA : array2 et (repA m k) { is_global gA })
  (gB : array2 et (repB k n) { is_global gB })
  (gC : array2 et (repC m n) { is_global gC })
  (rA rB rC : ematrix real _ _)
  (#eA #eB : ematrix _ _ _)
  (#eC : ematrix et m n)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB) **
    pure (tile /? m /\ tile /? n /\ tile /? k)
  requires
    pure (K.size_req m n k tile) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures (
    (exists* (eC' : ematrix et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB)))
{
  pts_to_ref_located gA;
  pts_to_ref_located gB;
  pts_to_ref_located gC;

  let mm = m /^ tile;
  let nn = n /^ tile;
  let kk = k /^ tile;

  K.mmcomb_gpu_approx tile
    comb comb_r
    #mm #nn #kk
    #(repA m k) #(repB k n) #(repC m n)
    #(crepA.inst m k)
    #(crepB.inst k n)
    #(crepC.inst m n)
    gA gB gC
    rA rB rC;

  ()
}

inline_for_extraction noextract
let spec_mm (tile : valid_tile) (et : Type0) {| scalar et, real_like et |} =
  spec tile et (fun _o n -> n) (fun _o n -> n)

inline_for_extraction noextract
let spec_gemm (tile : valid_tile) (et : Type0) {| scalar et, real_like et |}
  (repA repB repC : trepr2)
  {| crepA : ctrepr2 repA, crepB : ctrepr2 repB, crepC : ctrepr2 repC |}
  (alpha beta : et) =
  spec tile et
    (MS.lincomb alpha beta) (MS.lincomb (to_real alpha) (to_real beta))
    repA repB repC

(* dynamic tile *)
let g_matmul_f32_rrr tile = spec_mm tile f32 rm rm rm
let g_matmul_f64_rrr tile = spec_mm tile f64 rm rm rm
let g_matmul_u32_rrr tile = spec_mm tile u32 rm rm rm
let g_matmul_u64_rrr tile = spec_mm tile u64 rm rm rm

(* tile=32 *)
let g_matmul_f32_tile32_rrr = spec_mm 32sz f32 rm rm rm
let g_matmul_f64_tile32_rrr = spec_mm 32sz f64 rm rm rm
let g_matmul_u32_tile32_rrr = spec_mm 32sz u32 rm rm rm
let g_matmul_u64_tile32_rrr = spec_mm 32sz u64 rm rm rm

(* tile=16 *)
let g_matmul_f32_tile16_rrr = spec_mm 16sz f32 rm rm rm
let g_matmul_f64_tile16_rrr = spec_mm 16sz f64 rm rm rm
let g_matmul_u32_tile16_rrr = spec_mm 16sz u32 rm rm rm
let g_matmul_u64_tile16_rrr = spec_mm 16sz u64 rm rm rm

(* dynamic tile *)
let g_gemm_f32_rrr tile = spec_gemm tile f32 rm rm rm
let g_gemm_f64_rrr tile = spec_gemm tile f64 rm rm rm
let g_gemm_u32_rrr tile = spec_gemm tile u32 rm rm rm
let g_gemm_u64_rrr tile = spec_gemm tile u64 rm rm rm

(* tile=32 *)
let g_gemm_f32_tile32_rrr = spec_gemm 32sz f32 rm rm rm
let g_gemm_f64_tile32_rrr = spec_gemm 32sz f64 rm rm rm
let g_gemm_u32_tile32_rrr = spec_gemm 32sz u32 rm rm rm
let g_gemm_u64_tile32_rrr = spec_gemm 32sz u64 rm rm rm

(* tile=16 *)
let g_gemm_f32_tile16_rrr = spec_gemm 16sz f32 rm rm rm
let g_gemm_f64_tile16_rrr = spec_gemm 16sz f64 rm rm rm
let g_gemm_u32_tile16_rrr = spec_gemm 16sz u32 rm rm rm
let g_gemm_u64_tile16_rrr = spec_gemm 16sz u64 rm rm rm
