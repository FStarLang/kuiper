module KWitness8

(* Implementation of the imp8.cu witness. See KWitness8.fsti for the
   specification and the bit-equivalence rationale. *)

#lang-pulse
open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
module Alg = Kuiper.Tensor.Layout.Alg
module M = Kuiper.Tensor
module SZ = Kuiper.SizeT
module K = Kuiper.Kernel.GEMM.Naive1
module AB = Kuiper.Approximates.Base

(* The f32 combine `alpha*prod + beta*old` approximates its real analog. The
   add/mul approximation congruences need a nudge through the lambda. *)
let comb_approx2 (alpha beta : f32)
  : Lemma (Kuiper.Approximates.approx2
            (fun (v0 s : f32) -> (alpha `mul` s) `add` (beta `mul` v0))
            (fun (r0 rs : real) -> (to_real alpha *. rs) +. (to_real beta *. r0)))
  = introduce forall (x y : f32) (r s : real).
        (x %~ r /\ y %~ s ==>
          ((alpha `mul` y) `add` (beta `mul` x)) %~ ((to_real alpha *. s) +. (to_real beta *. r)))
    with introduce _ ==> _
    with _pf. (
      to_real_ok alpha;
      to_real_ok beta;
      AB.a_mul alpha y (to_real alpha) s;
      AB.a_mul beta x (to_real beta) r;
      AB.a_add (alpha `mul` y) (beta `mul` x) (to_real alpha *. s) (to_real beta *. r)
    )

fn matmul_f32
  (m n k : szp)
  (alpha beta : f32)
  (gA : M.array2 f32 (Alg.l2_row_major m k) { M.is_global gA })
  (gB : M.array2 f32 (Alg.l2_row_major k n) { M.is_global gB })
  (gC : M.array2 f32 (Alg.l2_row_major m n) { M.is_global gC })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#eA : chest2 f32 m k)
  (#eB : chest2 f32 k n)
  (#eC : chest2 f32 m n)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (SZ.v m * SZ.v n <= SZ.v max_blocks) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest2 f32 m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ gemm_real (to_real alpha) (to_real beta) rA rB (to_real_matrix eC)))
{
  comb_approx2 alpha beta;
  K.mmcomb_gpu_approx
    (fun v0 s -> (alpha `mul` s) `add` (beta `mul` v0))
    (fun r0 rs -> (to_real alpha *. rs) +. (to_real beta *. r0))
    gA gB gC
    rA rB (to_real_matrix eC);
  ()
}
