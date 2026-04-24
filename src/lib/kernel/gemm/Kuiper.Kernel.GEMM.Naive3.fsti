module Kuiper.Kernel.GEMM.Naive3

(* Like Naive2, but uses Kahan summation to reduce FP errors.
Thus only provides approximate spec. *)

#lang-pulse

open Kuiper
module T = Kuiper.Tensor
module M = Kuiper.Array2
module MS = Kuiper.Spec.GEMM
open Kuiper.EMatrix

inline_for_extraction noextract
let size_req : nat -> nat -> nat -> prop =
  fun m n k -> m * n <= max_blocks * max_threads

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n #k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA { M.is_global gA })
  (gB : M.array2 et lB { M.is_global gB })
  (gC : M.array2 et lC { M.is_global gC })
  (#eA : ematrix et m k)
  (#eB : ematrix et k n)
  (#eC : ematrix et m n)
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (rC : ematrix real m n)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req m n k) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : ematrix et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB)
