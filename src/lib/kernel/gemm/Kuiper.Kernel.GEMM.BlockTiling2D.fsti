module Kuiper.Kernel.GEMM.BlockTiling2D

#lang-pulse

open Kuiper
open Kuiper.Kernel.GEMMGPU.Type
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Array2 { array2 }
open Kuiper.Array2.Strided
open Kuiper.EMatrix
module M = Kuiper.Array2
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

(* Note: BlockTiling2D is the only tiled GEMM that has an exact spec
   (mmcomb_gpu_exact). This is so because it
   iterates through the shared dimension in the same left-to-right order
   as the pure mathematical product.

   The other tiled kernels (Tiled, SHMem, BlockTiling1D) accumulate
   partial results differently (e.g. via tiles that are added together)
   subproduct_cols), which introduces a different association order.

   We should probably rewrite the previous kernels to also
   attain an exact spec, though it is not a problem for now. *)

inline_for_extraction noextract
fn mmcomb_gpu_exact
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#rows #shared #cols : szp)
  (#lA : M.layout rows shared)
  (#lB : M.layout shared cols)
  (#lC : M.layout rows cols)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : array2 et lA { M.is_global gA })
  (#eA : ematrix et rows shared)
  (gB : array2 et lB { M.is_global gB })
  (#eB : ematrix et shared cols)
  (gC : array2 et lC { M.is_global gC })
  (#eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : M.full_layout bm bk)
  (slB : M.full_layout bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (M.core gA)) **
    pure (aligned 16 (M.core gB)) **
    pure (rows/bm * (cols/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#rows #shared #cols : szp)
  (#lA : M.layout rows shared)
  (#lB : M.layout shared cols)
  (#lC : M.layout rows cols)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : array2 et lA { M.is_global gA })
  (#eA : ematrix et rows shared)
  (gB : array2 et lB { M.is_global gB })
  (#eB : ematrix et shared cols)
  (gC : array2 et lC { M.is_global gC })
  (#eC : ematrix et rows cols)
  (bm : szp{bm /?+ rows})
  (bn : szp{bn /?+ cols})
  (bk : szp{bk /?+ shared})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : M.full_layout bm bk)
  (slB : M.full_layout bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (#fA #fB : perm)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (M.core gA)) **
    pure (aligned 16 (M.core gB)) **
    pure (rows/bm * (cols/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : ematrix et rows cols).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB)
