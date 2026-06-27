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
  (#m #n #k : szp)
  (gA : array2 et (rm m k) { is_global gA })
  (#fA : perm)
  (gB : array2 et (rm k n) { is_global gB })
  (#fB : perm)
  (gC : array2 et (rm m n) { is_global gC })
  (#eA : ematrix et m k)
  (#eB : ematrix et k n)
  (#eC : ematrix et m n)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (m * n <= max_blocks) **
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
  dguard (m   %^ bm = 0sz);
  dguard (k %^ bk = 0sz);
  dguard (n   %^ bn = 0sz);
  let mrows   = m /^ bm;
  let mshared = k /^ bk;
  let mcols   = n /^ bn;

  lemma_divides_trans (chunk et) bk k;
  assert pure (chunk et /?+ k);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v m) #(SZ.v k) (chunk et);

  lemma_divides_trans (chunk et) bn n;
  assert pure (chunk et /?+ n);
  lemma_aligned_strided_row_major_l2_row_major #(SZ.v k) #(SZ.v n) (chunk et);

  assert pure (SZ.fits (bm * bk));
  assert pure (SZ.fits (bk * bn));

  assert pure (m / bm <= m);
  assert pure (n / bn <= n);
  assert pure ((m / bm) * (n / bn) <= m * n);
  assert pure ((m / bm) * (n / bn) <= max_blocks);

  K.mmcomb_gpu_exact
    (fun o n -> mul beta o `add` mul alpha n)
    #m #n #k
    gA #eA gB #eB gC #eC
    bm bn bk
    tm tn
    (cm _ _) (rm _ _)
    #(Kuiper.Tensor.Layout.Alg.c_r2_col_major.inst bm bk)
    #(Kuiper.Tensor.Layout.Alg.c_r2_row_major.inst bk bn);

  ()
}
