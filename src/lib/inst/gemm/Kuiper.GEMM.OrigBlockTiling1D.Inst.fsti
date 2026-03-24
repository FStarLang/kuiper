module Kuiper.GEMM.OrigBlockTiling1D.Inst

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

module M = Kuiper.Matrix
module MU = Kuiper.Poly.GEMM.Util

inline_for_extraction noextract
fn spec
  (bm bn bk : szp)
  (tm : szp{tm /?+ bm /\ (bm/tm * bn <= max_threads)})
  (et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (rA rB rC : mrepr)
  {| cA : crepr rA, cB : crepr rB, cC :  crepr rC |}
  (rows shared cols : szp)
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (gA : M.gpu_matrix et (rA rows shared) { M.is_global_matrix gA })
  (#fA : perm)
  (gB : M.gpu_matrix et (rB shared cols) { M.is_global_matrix gB })
  (#fB : perm)
  (gC : M.gpu_matrix et (rC rows cols) { M.is_global_matrix gC })
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure ((rows/bm) * (cols/bn) <= max_blocks) **
    pure (bm/tm * bn <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : ematrix et _ _).
      on gpu_loc (gC |-> eC') **
      pure (eC' `ematrix_approximates` MU.real_mmcomb comb_r eC eA eB)
