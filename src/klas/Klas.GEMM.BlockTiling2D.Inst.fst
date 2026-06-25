module Klas.GEMM.BlockTiling2D.Inst

#lang-pulse
open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm, l2_col_major as cm }

module MS = Kuiper.Spec.GEMM
module K = Kuiper.Kernel.GEMM.BlockTiling2D
module SZ = Kuiper.SizeT

#set-options "--z3rlimit 100" // ridiculuous, try to improve

inline_for_extraction noextract
fn spec
  (bm bn bk : szp)
  (tm : szp{tm /? bm})
  (tn : szp{tn /? bn /\ (bm/tm * bn/tn <= max_threads)})
  (#_ : squash (sz_fits (bm*bk + (bm/tm * (bn/tn)))))
  (#_ : squash (sz_fits (bk*bn + (bm/tm * (bn/tn)))))
  (et : Type0) {| scalar et, has_vec_cpy et |}
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (alpha beta : et)
  (#rows #shared #cols : szp)
  (gA : array2 et (rm rows shared) { is_global gA })
  (#fA : perm)
  (gB : array2 et (rm shared cols) { is_global gB })
  (#fB : perm)
  (gC : array2 et (rm rows cols) { is_global gC })
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (rows * cols <= max_blocks) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.gemm alpha beta eC eA eB)
{
  tensor_pts_to_ref_located gA;
  tensor_pts_to_ref_located gB;
  tensor_pts_to_ref_located gC;

  // TODO: add dynamic assert for this.
    // pure (aligned 16 (core gA)) **
    // pure (aligned 16 (core gB)) **

  dassert (bm >^ 0sz);
  dassert (bn >^ 0sz);
  dassert (bk >^ 0sz);
  dassert (tm >^ 0sz);
  dassert (bm %^ tm = 0sz);
  dassert (bn %^ tn = 0sz);
  dguard (rows   %^ bm = 0sz);
  dguard (shared %^ bk = 0sz);
  dguard (cols   %^ bn = 0sz);
  let mrows   = rows   /^ bm;
  let mshared = shared /^ bk;
  let mcols   = cols   /^ bn;

  lemma_divides_trans (chunk et) bk shared;
  assert pure (chunk et /?+ shared);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v rows) #(SZ.v shared) (chunk et);

  lemma_divides_trans (chunk et) bn cols;
  assert pure (chunk et /?+ cols);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v shared) #(SZ.v cols) (chunk et);

  assert pure (SZ.fits (bm * bk));
  assert pure (SZ.fits (bk * bn));

  assert pure (rows / bm <= rows);
  assert pure (cols / bn <= cols);
  assert pure ((rows / bm) * (cols / bn) <= rows * cols);
  assert pure ((rows / bm) * (cols / bn) <= max_blocks);

  K.mmcomb_gpu_exact
    (fun o n -> mul beta o `add` mul alpha n)
    #rows #shared #cols
    gA #eA gB #eB gC #eC
    bm bn bk
    tm tn
    (cm _ _) (rm _ _)
    #(Kuiper.Tensor.Layout.Alg.c_r2_col_major.inst bm bk)
    #(Kuiper.Tensor.Layout.Alg.c_r2_row_major.inst bk bn);

  ()
}
