module Kuiper.GEMM.OrigBlockTiling1D.Inst

#lang-pulse
open Kuiper
open Kuiper.Approximates
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

module M = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Poly.GEMM.Util
module P = Kuiper.Poly.GEMM.OrigBlockTiling1D

inline_for_extraction noextract
fn spec
  (bm bn bk : szp)
  (tm : szp{tm /?+ bm /\ (bm/tm * bn <= max_threads)})
  (et : Type0) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { forall x y r s. x %~ r /\ y %~ s ==> comb x y %~ comb_r r s })
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
{
  M.gpu_matrix_pts_to_ref_located gA;
  M.gpu_matrix_pts_to_ref_located gB;
  M.gpu_matrix_pts_to_ref_located gC;

  dassert (bm >^ 0sz);
  dassert (bn >^ 0sz);
  dassert (bk >^ 0sz);
  dassert (tm >^ 0sz);
  dassert (bm %^ tm = 0sz);
  dguard (rows   %^ bm = 0sz);
  dguard (shared %^ bk = 0sz);
  dguard (cols   %^ bn = 0sz);
  // a HORRIBLE restriction
  dguard ((bm /^ tm *^ bn) = bm *^ bk);
  dguard ((bm /^ tm *^ bn) = bn *^ bk);
  dassert (bm = bn);
  let mrows   = rows   /^ bm;
  let mshared = shared /^ bk;
  let mcols   = cols   /^ bn;
  P.mmcomb_gpu_approx
    comb comb_r
    bm bn bk
    #(rows/^bm) #(shared/^bk) #(cols/^bn)
    tm
    #()
    #(rA rows shared) #(rB shared cols) #(rC rows cols)
    #(cA.map  _ _) #(cB.map _ _) #(cC.map _ _)
    gA gB gC
}
