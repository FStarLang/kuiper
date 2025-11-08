module Kuiper.GEMM.OrigBlockTiling1D.Inst

#lang-pulse
open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Reprs { row_major as rm, col_major as cm }

module M = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM

module P = Kuiper.Poly.GEMM.OrigBlockTiling1D

inline_for_extraction noextract
fn spec
  (bm bn bk : szp)
  (tm : szp{tm /?+ bm /\ (bm/tm * bn <= max_threads)})
  (et : Type0) {| scalar et |}
  (comb : binop et)
  (rA rB rC : mrepr)
  {| cA : crepr rA, cB : crepr rB, cC :  crepr rC |}
  (rows shared cols : szp)
  (#_: squash ((bm/tm * bn) == bm * bk /\ (bm/tm * bn) == bn * bk))
  (gA : M.gpu_matrix et (rA rows shared))
  (#fA : perm)
  (gB : M.gpu_matrix et (rB shared cols))
  (#fB : perm)
  (gC : M.gpu_matrix et (rC rows cols))
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure ((rows/bm) * (cols/bn) <= max_blocks) **
    pure (bm/tm * bn <= max_threads) **
    gC |-> eC
  ensures
    gC |-> MS.mmcomb comb eC eA eB
{
  M.gpu_matrix_pts_to_ref gA;
  M.gpu_matrix_pts_to_ref gB;
  M.gpu_matrix_pts_to_ref gC;

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

  P.mmcomb_gpu
    comb
    bm bn bk
    #(rows/^bm) #(shared/^bk) #(cols/^bn)
    tm
    #()
    #(rA rows shared) #(rB shared cols) #(rC rows cols)
    #(cA.map  _ _) #(cB.map _ _) #(cC.map _ _)
    gA gB gC;
}
