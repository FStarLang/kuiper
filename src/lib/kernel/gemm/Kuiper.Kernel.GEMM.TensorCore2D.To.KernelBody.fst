module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelBody

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.Float16
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorRO { vtlayout_of_tlayout }
module RO = Kuiper.TensorRO
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Copy.Vec2
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.Spec.GEMM
open Kuiper.TensorCore
open Pulse.Lib.Array
open Pulse.Lib.Trade

module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module T = Kuiper.Tensor
module FB = Kuiper.Kernel.GEMM.FlipFlopBarrier2
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.Epilogue
open Kuiper.Kernel.GEMM.TensorCore2D.To.KLoop
open Kuiper.Kernel.GEMM.TensorCore2D.To.Finish

#push-options "--split_queries no"
inline_for_extraction noextract
fn kf
  (#et_ab #et_cd #et_acc : Type0)
  {| scalar et_ab, has_vec_cpy et_ab,
     scalar et_cd, has_vec_cpy et_cd, scalar et_acc |}
  {| real_like et_ab, real_like et_cd, real_like et_acc |}
  (comb : et_cd -> et_acc -> et_cd)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n #k : szp)
  (#lA : layout2 m k) {| T.ctlayout lA |}
  (gA : array2 et_ab lA)
  (#eA : chest2 et_ab m k)
  (#lB : layout2 k n) {| T.ctlayout lB |}
  {| str_A : strided_row_major (vtlayout_of_tlayout lA),
     str_B : strided_row_major (vtlayout_of_tlayout lB) |}
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et_ab) str_B))
  (gB : array2 et_ab lB)
  (#eB : chest2 et_ab k n)
  (#lC : RO.vlayout2 m n)
  {| strC : strided_row_major lC |}
  (#_ : squash (aligned_strided_row_major (chunk et_cd) strC))
  (gC : RO.roarray2 et_cd lC)
  (#eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (#_ : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (#_ : squash (chunk et_ab /?+ bn))
  (#_ : squash (chunk et_ab /?+ bk))
  (#_ : squash (chunk et_cd /?+ n))
  (#_ : squash (chunk et_cd /?+ tn))
  (#_ : squash (SZ.fits (m * k)))
  (#_ : squash (SZ.fits (m * n)))
  (#_ : squash (SZ.fits (k * n)))
  (#_ : squash (SZ.fits (wm * tm)))
  (#_ : squash (SZ.fits (wn * tn)))
  (#_ : squash (SZ.fits (tm * tn + warp_size)))
  (#_ : squash (valid_frag_et_dims et_ab FragA tm tn tk))
  (#_ : squash (valid_frag_et_dims et_ab FragB tm tn tk))
  (#_ : squash (valid_frag_et_dims et_acc FragAcc tm tn tk))
  (#_ : squash (valid_frag_et_comb et_ab et_acc))
  (#fA #fB #fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (#_ : squash (warp_size /?+ nthr))
  (#_ : squash (SZ.fits ((nthr / warp_size) * tm * tn)))
  (#_ : squash (chunk et_ab * nthr /?+ (bm * bk)))
  (#_ : squash (chunk et_ab * nthr /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm * bk + nthr - 1)))
  (#_ : squash (SZ.fits (bk * bn + nthr - 1)))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (sh : c_shmems (shmems_desc_to et_ab et_acc bm bn bk tm tn nthr))
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  ()
  requires
    gpu **
    kpre_to gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn fA fB fC rA rB rC
      nblk nthr sh bid tid **
    thread_id nthr tid **
    block_id (m / bm * (n / bn)) bid **
    B.barrier_tok
      (FB.contract eA eB (rm bm bk) (rm bk bn)
        (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state 0
  ensures
    gpu **
    kpost_to comb_r gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn fA fB fC rA rB rC
      nblk nthr sh bid tid **
    thread_id nthr tid **
    block_id (m / bm * (n / bn)) bid **
    B.barrier_tok
      (FB.contract eA eB (rm bm bk) (rm bk bn)
        (fst sh) (fst (snd sh)) nthr bid) **
    B.barrier_state (2 * (k / bk))
{
  unfold kpre_to gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn fA fB fC rA rB rC
    nblk nthr sh bid tid;

  unfold kpre1_to gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn fA fB fC rA rB rC
    nblk nthr bid tid;
  let (sarA, (sarB, (sarAcc, sarTail))) = sh;

  let num_n_tiles = n /^ bn;
  let mrow = bid /^ num_n_tiles;
  assert pure (mrow < m / bm);
  let mcol = bid %^ num_n_tiles;
  assert pure (mcol < n / bn);
  let wid = tid /^ warp_size;
  let warpRow : szlt (bm / (wm * tm)) = wid /^ (bn /^ (wn *^ tn));
  let warpCol : szlt (bn / (wn * tn)) = wid %^ (bn /^ (wn *^ tn));
  Kuiper.Divides.lemma_div_product (wm * tm) bm m;
  FStar.Math.Lemmas.lemma_eucl_div_bound
    warpRow mrow (bm / (wm * tm));
  FStar.Math.Lemmas.lemma_mult_le_left
    (bm / (wm * tm)) (mrow + 1) (m / bm);
  assert pure (
    mrow * (bm / (wm * tm)) + warpRow < m / (wm * tm));
  let gwRow : enatlt (m / (wm * tm)) =
    mrow * (bm / (wm * tm)) + warpRow;
  Kuiper.Divides.lemma_div_product (wn * tn) bn n;
  FStar.Math.Lemmas.lemma_eucl_div_bound
    warpCol mcol (bn / (wn * tn));
  FStar.Math.Lemmas.lemma_mult_le_left
    (bn / (wn * tn)) (mcol + 1) (n / bn);
  assert pure (
    mcol * (bn / (wn * tn)) + warpCol < n / (wn * tn));
  let gwCol : enatlt (n / (wn * tn)) =
    mcol * (bn / (wn * tn)) + warpCol;

  let (sA, sB) = setup #et_ab #et_acc
    bm bn bk tm tn nthr (sarA, (sarB, (sarAcc, sarTail))) tid;

  let rAcc'' : chest2 real (wm * tm) (wn * tn) =
    MS.matmul
      (ematrix_subtile rA (wm * tm) k
        (warp_tile_i #m #n bm bn bk tm tn tk wm wn
          nthr bid (tid / warp_size)) 0)
      (ematrix_subtile rB k (wn * tn) 0
        (warp_tile_j #m #n bm bn bk tm tn tk wm wn
          nthr bid (tid / warp_size)));
  let accFrags : array (fragment et_acc FragAcc tm tn tk FragLAcc) =
    compute_acc #et_ab #et_acc gA gB
      bm bn bk tm tn tk wm wn rA rB nthr sA sB
      bid tid mrow mcol warpRow warpCol;
  let rOutTarget =
    ematrix_subtile
      (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
        bm bn mrow mcol)
      (wm * tm) (wn * tn) warpRow warpCol;
  prepare_epilogue #et_ab #et_cd #et_acc
    #_ #_ #_ #_ #_
    #m #n gC (fC /. (m / bm * (n / bn) * nthr)) eC rC
    bm bn bk tm tn tk wm wn nthr
    (sarA, (sarB, (sarAcc, sarTail))) accFrags rAcc'' tid;
  lemma_aligned_strided_row_major_l2_row_major
    #(SZ.v m) #(SZ.v n) #_ #_ (SZ.v (chunk et_cd));
  finish #et_ab #et_cd #et_acc comb comb_r
    gC gD
    bm bn bk tm tn tk wm wn
    rA rB rC nthr (sarA, (sarB, (sarAcc, sarTail))) sA sB
    bid tid mrow mcol wid warpRow warpCol gwRow gwCol
    accFrags rAcc'';
  rewrite each (SZ.v mrow) as (SZ.v bid / (SZ.v n / SZ.v bn));
  rewrite each (SZ.v mcol) as (SZ.v bid % (SZ.v n / SZ.v bn));
  rewrite each (SZ.v warpRow) as
    ((SZ.v tid / warp_size) / (SZ.v bn / (SZ.v wn * SZ.v tn)));
  rewrite each (SZ.v warpCol) as
    ((SZ.v tid / warp_size) % (SZ.v bn / (SZ.v wn * SZ.v tn)));
  fold kpost1_to comb_r gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn fA fB fC rA rB rC
    nblk nthr bid tid;
  rewrite
    shared_thread_live bm bn bk tm tn nthr
      (sarA, (sarB, (sarAcc, sarTail))) tid
  as
    shared_thread_live bm bn bk tm tn nthr sh tid;
  fold kpost_to comb_r gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn fA fB fC rA rB rC
    nblk nthr sh bid tid;
  ()
}
#pop-options
