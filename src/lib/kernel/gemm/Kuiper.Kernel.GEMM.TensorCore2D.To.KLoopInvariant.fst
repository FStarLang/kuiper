module Kuiper.Kernel.GEMM.TensorCore2D.To.KLoopInvariant

open Kuiper

#set-options "--z3rlimit 20 --fuel 1 --ifuel 1"

open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Spec.GEMM
open Kuiper.Tensor.Tiling

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc { constraints }

[@@"opaque_to_smt"]
let partial_matmul_step
  (#rows #shared #cols #tm #tk #tn : nat)
  (z : chest2 real tm tn)
  (a : chest2 (chest2 real tm tk) rows shared)
  (b : chest2 (chest2 real tk tn) shared cols)
  (row : natlt rows)
  (col : natlt cols)
  (vk : natlt shared)
  (rAcc : chest2 real tm tn {
    rAcc == tiled_partial_matmul z a b row col vk })
  (aTile : chest2 real tm tk { aTile == acc2 a row vk })
  (bTile : chest2 real tk tn { bTile == acc2 b vk col })
  : Lemma (
      rAcc `matplus` matmul aTile bTile
      == tiled_partial_matmul z a b row col (vk + 1))
=
  let lhs : chest2 real tm tn =
    rAcc `matplus` matmul aTile bTile in
  let rhs : chest2 real tm tn =
    tiled_partial_matmul z a b row col (vk + 1) in
  let aux (i : natlt tm) (j : natlt tn) :
    Lemma (acc2 lhs i j == acc2 rhs i j) =
    calc (==) {
      acc2 lhs i j;
      == {}
      acc2 (__gmatmul_single z matmul matplus a b row col vk
        `matplus` matmul (acc2 a row vk) (acc2 b vk col)) i j;
      == { __gmatmul_single_lemma
             z matmul matplus a b row col (vk + 1) }
      acc2 (__gmatmul_single z matmul matplus
        a b row col (vk + 1)) i j;
      == {}
      acc2 rhs i j;
    } in
  Classical.forall_intro_2 aux;
  assert (Kuiper.EMatrix.equal lhs rhs);
  ()

let loop_invariant_lemma
  (m n k : nat)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m))
  (#_ : squash (bn /?+ n))
  (#_ : squash (bk /?+ k))
  (mrow : natlt (m / bm))
  (mcol : natlt (n / bn))
  (warpRow : natlt (bm / (wm * tm)))
  (warpCol : natlt (bn / (wn * tn)))
  (gwRow : natlt (m / (wm * tm)) {
    gwRow == mrow * (bm / (wm * tm)) + warpRow })
  (gwCol : natlt (n / (wn * tn)) {
    gwCol == mcol * (bn / (wn * tn)) + warpCol })
  (vk : natlt (k / bk))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rAcc0 : chest2 real (wm * tm) (wn * tn) {
    rAcc0 == const _ 0.0R })
  (rAcc : chest2 real (wm * tm) (wn * tn))
  (#_ : squash (wm * tm /?+ m))
  (#_ : squash (wn * tn /?+ n))
  (#_ : squash (
    rAcc == tiled_partial_matmul rAcc0
      (ematrix_tiled rA (wm * tm) bk)
      (ematrix_tiled rB bk (wn * tn))
      gwRow gwCol vk))
  (rA_sub : chest2 real bm bk {
    rA_sub == ematrix_subtile rA bm bk mrow vk })
  (rB_sub : chest2 real bk bn {
    rB_sub == ematrix_subtile rB bk bn vk mcol })
  : Lemma (
      rAcc `matplus`
        matmul (ematrix_subtile rA_sub (wm * tm) bk warpRow 0)
               (ematrix_subtile rB_sub bk (wn * tn) 0 warpCol)
      == tiled_partial_matmul rAcc0
        (ematrix_tiled rA (wm * tm) bk)
        (ematrix_tiled rB bk (wn * tn))
        gwRow gwCol (vk + 1))
=
  let aux3 () :
    Lemma ((wm * tm) * gwRow == bm * mrow + warpRow * (wm * tm)) =
    calc (==) {
      (wm * tm) * gwRow;
      == {}
      (wm * tm) * (mrow * (bm / (wm * tm)) + warpRow);
      == { Math.Lemmas.distributivity_add_right
             (wm * tm) (mrow * (bm / (wm * tm))) warpRow }
      (wm * tm) * (mrow * (bm / (wm * tm))) + (wm * tm) * warpRow;
      == {}
      mrow * ((wm * tm) * (bm / (wm * tm))) + (wm * tm) * warpRow;
      == { Math.Lemmas.lemma_div_exact bm (wm * tm) }
      mrow * bm + (wm * tm) * warpRow;
      == {}
      bm * mrow + warpRow * (wm * tm);
    } in
  let aux4 () :
    Lemma ((wn * tn) * gwCol == bn * mcol + warpCol * (wn * tn)) =
    calc (==) {
      (wn * tn) * gwCol;
      == {}
      (wn * tn) * (mcol * (bn / (wn * tn)) + warpCol);
      == { Math.Lemmas.distributivity_add_right
             (wn * tn) (mcol * (bn / (wn * tn))) warpCol }
      (wn * tn) * (mcol * (bn / (wn * tn))) + (wn * tn) * warpCol;
      == {}
      mcol * ((wn * tn) * (bn / (wn * tn))) + (wn * tn) * warpCol;
      == { Math.Lemmas.lemma_div_exact bn (wn * tn) }
      mcol * bn + (wn * tn) * warpCol;
      == {}
      bn * mcol + warpCol * (wn * tn);
    } in
  aux3 ();
  aux4 ();
  let aux1 () : Lemma (
    ematrix_subtile rA_sub (wm * tm) bk warpRow 0
    == acc2 (ematrix_tiled rA (wm * tm) bk) gwRow vk)
  =
    macc_ematrix_tiled rA (wm * tm) bk gwRow vk;
    Kuiper.Chest.ext
      (ematrix_subtile rA_sub (wm * tm) bk warpRow 0)
      (acc2 (ematrix_tiled rA (wm * tm) bk) gwRow vk) in
  let aux2 () : Lemma (
    ematrix_subtile rB_sub bk (wn * tn) 0 warpCol
    == acc2 (ematrix_tiled rB bk (wn * tn)) vk gwCol)
  =
    macc_ematrix_tiled rB bk (wn * tn) vk gwCol;
    Kuiper.Chest.ext
      (ematrix_subtile rB_sub bk (wn * tn) 0 warpCol)
      (acc2 (ematrix_tiled rB bk (wn * tn)) vk gwCol) in
  aux1 ();
  aux2 ();
  partial_matmul_step rAcc0
    (ematrix_tiled rA (wm * tm) bk)
    (ematrix_tiled rB bk (wn * tn))
    gwRow gwCol vk rAcc
    (ematrix_subtile rA_sub (wm * tm) bk warpRow 0)
    (ematrix_subtile rB_sub bk (wn * tn) 0 warpCol)
