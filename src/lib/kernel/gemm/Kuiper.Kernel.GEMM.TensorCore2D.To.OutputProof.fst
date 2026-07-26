module Kuiper.Kernel.GEMM.TensorCore2D.To.OutputProof

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1"
#set-options "--z3rlimit 15"

open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc
open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

noextract
ghost fn normalize_output
  (#et_cd : Type0) {| scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ m /\ wn * tn /?+ n))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (bid : szlt (m / bm * (n / bn)))
  (tid : szlt nthr)
  (mrow : szlt (m / bm))
  (mcol : szlt (n / bn))
  (wid : szlt (nthr / warp_size))
  (warpRow : szlt (bm / (wm * tm)))
  (warpCol : szlt (bn / (wn * tn)))
  (#_ : squash (
    SZ.v mrow == SZ.v bid / (SZ.v n / SZ.v bn) /\
    SZ.v mcol == SZ.v bid % (SZ.v n / SZ.v bn) /\
    SZ.v wid == SZ.v tid / warp_size /\
    SZ.v warpRow == SZ.v wid / (SZ.v bn / (SZ.v wn * SZ.v tn)) /\
    SZ.v warpCol == SZ.v wid % (SZ.v bn / (SZ.v wn * SZ.v tn))))
  (gwRow : enatlt (m / (wm * tm)) {
    gwRow == mrow * (bm / (wm * tm)) + warpRow })
  (gwCol : enatlt (n / (wn * tn)) {
    gwCol == mcol * (bn / (wn * tn)) + warpCol })
  (rAcc : chest2 real (wm * tm) (wn * tn) {
    rAcc == MS.matmul
      (ematrix_subtile rA (wm * tm) k
        (warp_tile_i #m #n bm bn bk tm tn tk wm wn
          nthr bid (tid / warp_size)) 0)
      (ematrix_subtile rB k (wn * tn) 0
        (warp_tile_j #m #n bm bn bk tm tn tk wm wn
          nthr bid (tid / warp_size))) })
  requires
    output_lane_approximates gD bm bn tm tn wm wn bid tid
      (chest_comb comb_r
        (ematrix_subtile
          (ematrix_subtile rC bm bn
            (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn))))
        rAcc)
  ensures
    output_lane_approximates gD bm bn tm tn wm wn bid tid
      (ematrix_subtile
        (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
          bm bn mrow mcol)
        (wm * tm) (wn * tn) warpRow warpCol)
{
  MS.matmul_decompose_lemma rA rB
    (wm * tm) (wn * tn) gwRow gwCol;
  let rOutLocal =
    chest_comb comb_r
      (ematrix_subtile
        (ematrix_subtile rC bm bn mrow mcol)
        (wm * tm) (wn * tn) warpRow warpCol)
      rAcc;
  let rOutTarget =
    ematrix_subtile
      (ematrix_subtile
        (MS.mmcomb comb_r rC rA rB)
        bm bn mrow mcol)
      (wm * tm) (wn * tn) warpRow warpCol;
  rewrite each
    (SZ.v bid / (SZ.v n / SZ.v bn))
  as SZ.v mrow;
  rewrite each
    (SZ.v bid % (SZ.v n / SZ.v bn))
  as SZ.v mcol;
  rewrite each
    (SZ.v tid / warp_size)
  as SZ.v wid;
  rewrite each
    (SZ.v wid / (SZ.v bn / (SZ.v wn * SZ.v tn)))
  as SZ.v warpRow;
  rewrite each
    (SZ.v wid % (SZ.v bn / (SZ.v wn * SZ.v tn)))
  as SZ.v warpCol;
  rewrite each
    (chest_comb comb_r
      (ematrix_subtile
        (ematrix_subtile rC bm bn mrow mcol)
        (wm * tm) (wn * tn) warpRow warpCol)
      rAcc)
  as rOutLocal;
  assert pure (
    (SZ.v bm / (SZ.v wm * SZ.v tm)) *
      (SZ.v wm * SZ.v tm) == SZ.v bm);
  assert pure (
    (SZ.v bn / (SZ.v wn * SZ.v tn)) *
      (SZ.v wn * SZ.v tn) == SZ.v bn);
  assert pure (
    gwRow * (SZ.v wm * SZ.v tm) ==
    SZ.v mrow * SZ.v bm +
      SZ.v warpRow * (SZ.v wm * SZ.v tm));
  assert pure (
    gwCol * (SZ.v wn * SZ.v tn) ==
    SZ.v mcol * SZ.v bn +
      SZ.v warpCol * (SZ.v wn * SZ.v tn));
  Kuiper.EMatrix.lemma_equal_intro rOutLocal rOutTarget;
  Kuiper.Chest.ext rOutLocal rOutTarget;
  rewrite
    output_lane_approximates
      gD bm bn tm tn wm wn bid tid rOutLocal
  as
    output_lane_approximates
      gD bm bn tm tn wm wn bid tid rOutTarget;
  rewrite each rOutTarget as
    ematrix_subtile
      (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
        bm bn mrow mcol)
      (wm * tm) (wn * tn) warpRow warpCol;
}
