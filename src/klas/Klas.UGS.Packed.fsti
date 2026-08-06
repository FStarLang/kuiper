module Klas.UGS.Packed
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.EMatrix

open Klas.UGS.WeightViews
open Kuiper.Kernel.UGS.Epilogue

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

(* Relational adapter for wmma_ugs.cu's physical weight allocation.
   Semantically this still takes A, W_gate, and W_up as three matrices;
   the two weight pointers may alias because their verified layouts select
   disjoint cells from the same packed allocation. *)
fn swiglu
  (m k n : szp)
  (#_ : squash (n % pair_group_n == 0))
  (#_ : squash (SZ.fits (2 * n)))
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (k * (2 * n))))
  (#_ : squash (SZ.fits (m * n)))
  (#_ : squash (m % 64 == 0))
  (#_ : squash (k % 16 == 0))
  (#_ : squash (n % 64 == 0))
  (#_ : squash ((m / 64) * (n / 64) <= max_blocks))
  (gA : array2 half (rm m k) { is_global gA })
  (gWGate : array2 half (gate_layout k n) { is_global gWGate })
  (gWUp : array2 half (up_layout k n) { is_global gWUp })
  (gOut : array2 half (rm m n) { is_global gOut })
  (#_ : squash (aligned 16 (core gA)))
  (#eA : chest2 half m k)
  (#eWGate #eWUp : chest2 half k n)
  (#eOut0 : chest2 half m n)
  (#fA #fWGate #fWUp : perm)
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gWGate |-> Frac fWGate eWGate) **
    on gpu_loc (gWUp |-> Frac fWUp eWUp)
  requires
    on gpu_loc (gOut |-> eOut0)
  ensures
    (exists* (eOut : chest2 half m n).
      on gpu_loc (gOut |-> eOut) **
      pure (eOut %~ chest_comb real_swiglu
        (MS.matmul (to_real_matrix eA) (to_real_matrix eWGate))
        (MS.matmul (to_real_matrix eA) (to_real_matrix eWUp))))
