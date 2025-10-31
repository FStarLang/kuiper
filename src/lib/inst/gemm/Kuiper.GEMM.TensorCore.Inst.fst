module Kuiper.GEMM.TensorCore.Inst
#lang-pulse

open Kuiper
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs
open Kuiper.TensorCore

module SZ = Kuiper.SizeT

open Kuiper.Poly.GEMM.TensorCore
#set-options "--z3rlimit 20"

inline_for_extraction noextract
fn specialize_gpu
  // specialize
  (et_ab et_c : Type0)
  {| scalar et_ab, scalar et_c |}
  (bm bn bk : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn})
  (tk : szp{tk /? bk})
  // should be up here! if part of the precondition, then
  //  the value is not checked for correctness when
  //  the function is only partially applied!
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_c FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_c))
  (#_ : squash (SZ.fits (bm * bk)))
  (#_ : squash (SZ.fits (bk * bn)))
  (#_ : squash (bm/tm * bn/tn * warp_size <= max_threads))
  (#_ : squash (SZ.fits (bm*bk + bm/tm * bn/tn * warp_size)))
  (#_ : squash (SZ.fits (bk*bn + bm/tm * bn/tn * warp_size)))
  (rA rB rC : mrepr)
  {| ca : crepr rA, cB : crepr rB, cC : crepr rC |}

  // do not specialize
  (rows shared cols : szp)
  (gA : gpu_matrix et_ab (rA rows shared))
  (gB : gpu_matrix et_ab (rB shared cols))
  (gC : gpu_matrix et_c (row_major rows cols))
  (#eA : ematrix et_ab rows shared)
  (#eB : ematrix et_ab shared cols)
  (#eC : ematrix et_c rows cols)
  (#fA #fB : perm)
  // non of these are are checked because the functions is only
  //  partially applied
  preserves
    cpu **
    // should be checked at runtime
    pure (rows * cols <= max_blocks) **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    gC |-> eC
  ensures
    (exists* eC'. gC |-> eC')
{
  gpu_matrix_pts_to_ref gA;
  gpu_matrix_pts_to_ref gB;
  gpu_matrix_pts_to_ref gC;

  let mrows   = rows   /^ bm;
  let mshared = shared /^ bk;
  let mcols   = cols   /^ bn;

  // preconditions checcked at runtime
  // TODO should be checked at runtime but has ghost effect:
  //  dguard (SZ.lte (rows *^ cols) (SZ.uint_to_t max_blocks));
  dguard (rows   %^ bm = 0sz);
  dguard (shared %^ bk = 0sz);
  dguard (cols   %^ bn = 0sz);

  let nblk = rows/^bm *^ (cols/^bn);
  let nthr = bm/^tm *^ (bn/^tn) *^ warp_sz;
  launch_sync (
    mk_kernel gA gB gC bm bn bk tm tn tk nblk nthr ()
  );

  ()
}
