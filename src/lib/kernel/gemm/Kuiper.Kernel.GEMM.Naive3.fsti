module Kuiper.Kernel.GEMM.Naive3

(* Like Naive2, but uses Kahan summation to reduce FP errors.
Thus only provides approximate spec. *)

#lang-pulse

open Kuiper
module MS = Kuiper.Spec.GEMM
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.Chest
open Kuiper.Kernel.GEMMGPU.Type

inline_for_extraction noextract
let size_req : nat -> nat -> nat -> prop =
  fun m n k -> m * n <= max_blocks * max_threads

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (gA : tensor et lA { is_global gA })
  (gB : tensor et lB { is_global gB })
  (gC : tensor et lC { is_global gC })
  (#eA : chest2 et m k)
  (#eB : chest2 et k n)
  (#eC : chest2 et m n)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB)
