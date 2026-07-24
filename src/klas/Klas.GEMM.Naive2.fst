module Klas.GEMM.Naive2

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util
module K = Kuiper.Kernel.GEMM.Naive2
module C = Kuiper.Matrix.Casts
module SZ = Kuiper.SizeT

(* Specialize *)
inline_for_extraction noextract
fn spec
  (et : Type0) {| scalar et, real_like et |}
  (batch m n k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (rA rB : chest3 real _ _ _)
  (#eA #eB #eC : chest3 _ _ _ _)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (K.bsize_req batch m n k) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.batched_matmul eA eB) **
    pure (MS.batched_matmul eA eB %~ MS.batched_matmul rA rB)
{
  K.bmmcomb_gpu_exact MS.comb2 batch m n k gA gB gC;
  MU.bmmcomb_approx_real
    MS.comb2 MS.comb2
    eA eB eC
    rA rB (Kuiper.Chest.to_real_chest eC);
  ()
}

(* Lowering a one-page batched matmul yields the rank-2 matmul. *)
let batch1_batched_matmul
  (#et : Type0) {| scalar et |}
  (rows shared cols : szp)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  : Lemma (
      C.c3_to_c2 rows cols
        (MS.batched_matmul (C.c2_to_c3 rows shared eA) (C.c2_to_c3 shared cols eB))
      == MS.matmul eA eB)
  =
  C.c2_to_c3_slice_page rows shared eA;
  C.c2_to_c3_slice_page shared cols eB;
  assert (equal
      (C.c3_to_c2 rows cols
         (MS.batched_matmul (C.c2_to_c3 rows shared eA) (C.c2_to_c3 shared cols eB)))
        (MS.matmul eA eB))

(* Rank-2 spec *)
inline_for_extraction noextract
fn spec_2d
  (et : Type0) {| scalar et, real_like et |}
  (repA repB repC : trepr2)
  {| ctrepr2 repA, ctrepr2 repB, ctrepr2 repC |}
  (m n k : szp)
  (gA : tensor et (repA m k) { is_global gA })
  (gB : tensor et (repB k n) { is_global gB })
  (gC : tensor et (repC m n) { is_global gC })
  (rA rB : chest2 real _ _)
  (#eA #eB #eC : chest2 _ _ _)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (m * n <= max_blocks) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures (
    (exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul rA rB)))
{
  map_loc gpu_loc (fun () -> C.t2_to_t3 m k gA);
  map_loc gpu_loc (fun () -> C.t2_to_t3 k n gB);
  map_loc gpu_loc (fun () -> C.t2_to_t3 m n gC);

  spec et
    1sz m n k
    (relay gA (C.l2_to_l3 m k #(repA m k)))
    (relay gB (C.l2_to_l3 k n #(repB k n)))
    (relay gC (C.l2_to_l3 m n #(repC m n)))
    (C.c2_to_c3 m k rA)
    (C.c2_to_c3 k n rB);

  map_loc gpu_loc (fun () -> C.t3_to_t2 m k gA);
  map_loc gpu_loc (fun () -> C.t3_to_t2 k n gB);
  map_loc gpu_loc (fun () -> C.t3_to_t2 m n gC);

  C.c2_to_c3_roundtrip m k eA;

  C.c2_to_c3_roundtrip k n eB;

  batch1_batched_matmul m k n rA rB;
  ()
}

let g_matmul_f32_rrr = spec_2d f32 l2_row_major l2_row_major l2_row_major
let g_matmul_f64_rrr = spec_2d f64 l2_row_major l2_row_major l2_row_major
let g_matmul_u32_rrr = spec_2d u32 l2_row_major l2_row_major l2_row_major
let g_matmul_u64_rrr = spec_2d u64 l2_row_major l2_row_major l2_row_major

let g_matmul_f32_ccc = spec_2d f32 l2_col_major l2_col_major l2_col_major
let g_matmul_f64_ccc = spec_2d f64 l2_col_major l2_col_major l2_col_major
let g_matmul_u32_ccc = spec_2d u32 l2_col_major l2_col_major l2_col_major
let g_matmul_u64_ccc = spec_2d u64 l2_col_major l2_col_major l2_col_major

let batched_matmul_f32 (batch m n k : szp)
  (#_ : (SZ.fits (batch * m * k) /\ SZ.fits (batch * k * n) /\ SZ.fits (batch * m * n)))
  =
  K.bmmcomb_gpu_exact #f32 MS.comb2 batch m n k
    #(l3_batched_row_major _ _ _)
    #(l3_batched_row_major _ _ _)
    #(l3_batched_row_major _ _ _)

let batched_gemm_f32 alpha beta (batch m n k : szp)
  (#_ : (SZ.fits (batch * m * k) /\ SZ.fits (batch * k * n) /\ SZ.fits (batch * m * n))) =
  K.bmmcomb_gpu_exact #f32 (MS.lincomb alpha beta) batch m n k
    #(l3_batched_row_major _ _ _)
    #(l3_batched_row_major _ _ _)
    #(l3_batched_row_major _ _ _)
