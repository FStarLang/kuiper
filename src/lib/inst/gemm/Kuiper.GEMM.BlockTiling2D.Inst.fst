module Kuiper.GEMM.BlockTiling2D.Inst

#lang-pulse
open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Reprs { row_major as rm, col_major as cm }

module M = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module P = Kuiper.Poly.GEMM.BlockTiling2D
module SZ = FStar.SizeT

#set-options "--z3rlimit 30"

inline_for_extraction noextract
fn spec
  (bm bn bk : szp)
  (slA : full_mlayout bm bk)
  (slB : full_mlayout bk bn)
  {| clayout slA, clayout slB |}
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn /\ (bm/tm * bn/tn <= max_threads)})
  (#_ : squash (SZ.fits (bm*bk + (bm/tm * (bn/tn)))))
  (#_ : squash (SZ.fits (bk*bn + (bm/tm * (bn/tn)))))
  (et : Type0) {| scalar et, has_vec_cpy et |}
  (#_ : squash (chunk et /? bn))
  (#_ : squash (chunk et /? bk))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /? (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /? (bk * bn)))
  (alpha beta : et)
  (#rows #shared #cols : szp)
  (gA : M.gpu_matrix et (rm rows shared))
  (#fA : perm)
  (gB : M.gpu_matrix et (rm shared cols))
  (#fB : perm)
  (gC : M.gpu_matrix et (rm rows cols))
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (rows * cols <= max_blocks) **
    gC |-> eC
  ensures
    gC |-> MS.gemm alpha beta eC eA eB
{
  // These didn't seem to be needed before!?
  M.gpu_matrix_pts_to_ref gA;
  M.gpu_matrix_pts_to_ref gB;
  M.gpu_matrix_pts_to_ref gC;

  dassert (bm `SZ.gt` 0sz);
  dassert (bn `SZ.gt` 0sz);
  dassert (bk `SZ.gt` 0sz);
  dassert (tm `SZ.gt` 0sz);
  dassert (bm %^ tm = 0sz);
  dassert (bn %^ tn = 0sz);
  dguard (rows   %^ bm = 0sz);
  dguard (shared %^ bk = 0sz);
  dguard (cols   %^ bn = 0sz);
  let mrows   = rows   /^ bm;
  let mshared = shared /^ bk;
  let mcols   = cols   /^ bn;

  P.mmcomb_gpu
    (fun o n -> mul beta o `add` mul alpha n)
    #rows #shared #cols
    gA #eA gB #eB gC #eC
    bm bn bk
    tm tn
    slA slB #_ #_;

  ()
}
