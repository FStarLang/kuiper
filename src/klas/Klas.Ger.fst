module Klas.Ger

(* cuBLAS ger / geru: rank-1 update  A := alpha * x * y^T + A.

   This is exactly the alpha/beta GEMM with the inner dimension k = 1 and
   beta = 1: A (m x 1, i.e. the column vector x) times B (1 x n, i.e. the row
   vector y^T) is the outer product x*y^T, and beta = 1 accumulates into A.
   We therefore reuse the verified naive GEMM kernel (Klas.GEMM.NaiveAB) and
   only specialize the two scalars, so the spec is exact at the element type.

   The generated C entry point takes (m, n, alpha, x, y, A) device pointers.
   Transpose / conjugate variants are out of scope (the model has no complex
   type and selects A's transpose via the chosen matrix layout). *)

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module NB = Klas.GEMM.NaiveAB

inline_for_extraction noextract
fn ger_spec
  (et : Type0) {| scalar et, real_like et |}
  (repA repB repC : trepr2)
  {| ctrepr2 repA, ctrepr2 repB, ctrepr2 repC |}
  (m n : szp)
  (alpha : et)
  (gA : tensor et (repA m 1sz) { is_global gA })   (* x : m x 1 *)
  (gB : tensor et (repB 1sz n) { is_global gB })   (* y^T : 1 x n *)
  (gC : tensor et (repC m n) { is_global gC })      (* A : m x n *)
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
    exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb (MS.rlincomb (to_real alpha) (to_real (one #et)))
                            (Kuiper.Chest.to_real_chest eC) rA rB))
{
  NB.spec et repA repB repC m n 1sz alpha one gA gB gC rA rB;
}

(* x, y, A all stored contiguously (row-major); for a 1-column / 1-row / m x n
   matrix this is just the natural dense vector / matrix layout. *)
let ger_f32 = ger_spec f32 l2_row_major l2_row_major l2_row_major
let ger_f64 = ger_spec f64 l2_row_major l2_row_major l2_row_major
let ger_u32 = ger_spec u32 l2_row_major l2_row_major l2_row_major
let ger_u64 = ger_spec u64 l2_row_major l2_row_major l2_row_major
