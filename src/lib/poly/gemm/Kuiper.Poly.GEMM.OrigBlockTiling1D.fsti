module Kuiper.Poly.GEMM.OrigBlockTiling1D

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Poly.GEMMGPU.Type
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
module MU = Kuiper.Poly.GEMM.Util

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
  (bm bn bk : szp)
  (#mrows #mshared #mcols : szp)
  (tm : szp{tm /?+ bm})
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (#lA : mlayout (mrows   * bm) (mshared * bk))
  (#lB : mlayout (mshared * bk) (mcols   * bn))
  (#lC : mlayout (mrows   * bm) (mcols   * bn))
  {| clayout lA, clayout lB, clayout lC |}
  (gA : gpu_matrix et lA { is_global_matrix gA })
  (#fA : perm)
  (gB : gpu_matrix et lB { is_global_matrix gB })
  (#fB : perm)
  (gC : gpu_matrix et lC { is_global_matrix gC })
  (#eA : ematrix et (mrows   * bm) (mshared * bk))
  (#eB : ematrix et (mshared * bk) (mcols   * bn))
  (#eC : ematrix et (mrows   * bm) (mcols   * bn))
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (mrows * mcols <= max_blocks) **
    pure (bm/tm * bn <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et _ _).
      on gpu_loc (gC |-> eC') **
      pure (ematrix_approximates eC' (MU.real_mmcomb comb_r eC eA eB)))
