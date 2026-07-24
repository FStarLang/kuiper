module Klas.GEMM.TensorCore2D.To.Inst
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.TensorCore
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Float.Casts { float_cast }
module MS = Kuiper.Spec.GEMM

module SZ = Kuiper.SizeT
module K = Kuiper.Kernel.GEMM.TensorCore2D.To

#push-options "--split_queries always --z3rlimit 40"
inline_for_extraction noextract
fn spec
  // specialize
  (et_ab et_acc et_cd : Type0)
  {| scalar et_ab, has_vec_cpy et_ab, real_like et_ab |}
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, has_vec_cpy et_cd, real_like et_cd |}
  {| float_cast et_cd et_acc, float_cast et_acc et_cd |}
  (bm bn bk : szp)
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_ab /?+ bn))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (tk : szp{tk /?+ bk})
  (wm : szp{wm * tm /?+ bm})
  (wn : szp{wn * tn /?+ bn})
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (wm * wn)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#_ : squash (SZ.fits (bm*bk + (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) - 1)))
  (#_ : squash (SZ.fits (bk*bn + (bm/(wm*tm) * (bn/(wn*tn)) * warp_size) - 1)))
  (#_ : squash ((bm/(wm*tm) * (bn/(wn*tn)) * (SZ.v warp_size)) <= max_threads))

  // do not specialize
  (rows shared cols : szp)
  (gA : array2 et_ab (rm rows shared) { is_global gA })
  (gB : array2 et_ab (rm shared cols) { is_global gB })
  (gC : array2 et_cd (rm rows cols) { is_global gC })
  (gD : array2 et_cd (rm rows cols) { is_global gD })
  (alpha beta : et_acc)
  (#_ : squash (aligned 16 (core gA)))
  (#_ : squash (aligned 16 (core gB)))
  (#eA : chest2 et_ab rows shared)
  (#eB : chest2 et_ab shared cols)
  (#eC : chest2 et_cd rows cols)
  (#fA #fB #fC : perm)
  preserves
    cpu **
    pure ((rows/bm) * (cols/bn) <= max_blocks) **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (gC |-> Frac fC eC)
  requires
    pure (SZ.fits (rows * cols)) **
    on gpu_loc (live gD)
  ensures
    exists* eD'.
      on gpu_loc (gD |-> eD') **
      pure (eD' %~ MS.mmcomb
        (MS.rlincomb (to_real alpha) (to_real beta))
        (to_real_matrix eC)
        (to_real_matrix eA)
        (to_real_matrix eB))
{
  tensor_pts_to_ref_located gA;
  tensor_pts_to_ref_located gB;
  tensor_pts_to_ref_located gC;
  tensor_pts_to_ref_located gD;

  dassert (bm %^ tm = 0sz);
  dassert (bn %^ tn = 0sz);
  dassert (bk %^ tk = 0sz);

  dguard (rows   %^ bm = 0sz);
  dguard (shared %^ bk = 0sz);
  dguard (cols   %^ bn = 0sz);

  lemma_divides_chain (wm * tm) bm rows;
  lemma_divides_chain (wn * tn) bn cols;

  let nblk = rows/^bm *^ (cols/^bn);
  let nthr = bm/^(wm*^tm) *^ (bn/^(wn*^tn)) *^ warp_size;

  assert pure ((rows/bm) * (cols/bn) == nblk);
  assert pure ((rows/bm) * (cols/bn) <= max_blocks);
  dassert (nblk <=^ SZ.uint_to_t 2097152);
  assert pure (nblk <= max_blocks);

  dassert ((bm *^ bk) %^ (chunk et_ab *^ nthr) = 0sz);
  dassert ((bk *^ bn) %^ (chunk et_ab *^ nthr) = 0sz);

  lemma_divides_trans (chunk et_ab) bk shared;
  assert pure (chunk et_ab /?+ shared);
  lemma_aligned_strided_row_major_l2_row_major
    #(SZ.v rows) #(SZ.v shared) (chunk et_ab);

  lemma_divides_trans (chunk et_ab) bn cols;
  assert pure (chunk et_ab /?+ cols);
  lemma_aligned_strided_row_major_l2_row_major
    #(SZ.v shared) #(SZ.v cols) (chunk et_ab);

  let rA = to_real_matrix eA;
  let rB = to_real_matrix eB;
  let rC = to_real_matrix eC;
  let comb =
    (fun (x : et_cd) (y : et_acc) ->
      MS.lincomb_to #et_acc #et_cd alpha beta x y);
  let comb_r = MS.rlincomb (to_real alpha) (to_real beta);
  MS.lincomb_to_approx2 #et_acc #et_cd alpha beta;

  #set-options "--fuel 0 --ifuel 0 --z3refresh" {
  launch_sync (
    K.mk_kernel comb comb_r
      gA #eA gB #eB
      gC #_ #eC gD #eC
      bm bn bk tm tn tk wm wn
      #_ #_ #_ #_ #_ #_ #_ #_
      #fA #fB #fC
      nblk nthr
      #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_ #_
      rA rB rC #_ #_ ()
  )};

  ()
}
#pop-options
