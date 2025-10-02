module Kuiper.Poly.GEMM.Util

#lang-pulse

open Kuiper
open Pulse.Lib.Trade
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module SZ = FStar.SizeT
module Tiling = Kuiper.Matrix.Tiling
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

inline_for_extraction noextract
fn matmul_dotprod
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : SZ.t)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  {| clayout lA, clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (i : szlt rows)
  (j : szlt cols)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res == MS.matmul_single eA eB i j)
{
  let mut k : sz = 0sz;
  let mut sum : et = zero;

  while (SZ.(!k <^ shared))
    invariant
      exists* (vk : SZ.t{vk <= shared}).
        k |-> vk **
        sum |-> MS.__matmul_single eA eB i j vk
  {
    let v1 = M.gpu_matrix_read gA i !k;
    let v2 = M.gpu_matrix_read gB !k j;

    let vsum = !sum;
    sum := vsum `add` mul v1 v2;
    k := SZ.add !k 1sz;

    (**)MS.matmul_single_lemma eA eB i j !k;
    ();
  };
  !sum
}

inline_for_extraction noextract
fn matmul_tiled_dotprod
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : sz)
  (#tile : szp)
  (#lA : mlayout (rows   * tile) (shared * tile))
  (#lB : mlayout (shared * tile) (cols   * tile))
  {| clayout lA, clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA #eB : ematrix _ _ _)
  (bi : szlt rows)
  (bj : szlt cols)
  (i : szlt tile)
  (j : szlt tile)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  // ensures
  //   pure (res == MS.matmul_single #et #_ #(rows * tile) #(shared * tile) #(cols * tile) eA eB i j shared)
{
  let mut sum : et = zero;
  let mut bk  : sz = 0sz;

  while (SZ.(!bk <^ shared))
    invariant
      exists* (vbk : SZ.t) sumv.
        pure (vbk <= shared) **
        bk |-> vbk **
        sum |-> sumv
  {
    let vbk = !bk;
    let s = !sum;
    assert (pure (bi  < (rows   * tile) / tile));
    assert (pure (vbk < (shared * tile) / tile));
    // Sigh.... need to reveal and hide. Terrible UX.
    let tA = Tiling.gpu_matrix_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v bi) (SZ.v vbk);
    let tB = Tiling.gpu_matrix_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v vbk) (SZ.v bj);
    assert (rewrites_to tA (Tiling.gpu_matrix_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v bi) (SZ.v vbk)));
    assert (rewrites_to tB (Tiling.gpu_matrix_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v vbk) (SZ.v bj)));

    Tiling.gpu_matrix_extract_tile_ro gA tile tile bi vbk;
    Tiling.gpu_matrix_extract_tile_ro gB tile tile vbk bj;

    let s' = matmul_dotprod tA tB i j;
    sum := !sum `add` s';

    ambig_trade_elim ();
    ambig_trade_elim ();

    bk := !bk +^ 1sz;
  };

  !sum
}

(* Used by SHMEM, Blocktiling1D *)
inline_for_extraction noextract
fn subproduct_cols
  (#et : Type0) {| scalar et |}
  (tile : sz)
  (acc : array et)
  (#l1 : mlayout tile tile) {| clayout l1 |}
  (#l2 : mlayout tile tile) {| clayout l2 |}
  (m1 : M.gpu_matrix et l1)
  (m2 : M.gpu_matrix et l2)
  (j : szlt tile)
  (#acc0 : erased (seq et))
  (#v1 #v2 : ematrix et tile tile)
  (#f : perm)
  preserves
    gpu **
    m1 |-> Frac f v1 **
    m2 |-> Frac f v2
  requires
    pure (Seq.length acc0 == tile) **
    acc |-> acc0
  ensures
    exists* acc'.
      pure (Seq.length acc' == tile) **
      (acc |-> acc')
{
  pts_to_len acc;
  let mut sk : sz = 0sz;
  while (SZ.(!sk <^ tile))
    invariant live sk ** live acc
  {
    pts_to_len acc;
    let mut i = 0sz;
    (* We can read v2 out of the inner loop, this is extremely
       important for performance. NVCC may realize this is invariant
       across iterations and hoist it out, but don't rely on it. *)
    let v2 = M.gpu_matrix_read m2 !sk j;
    while (SZ.(!i <^ tile))
      invariant live i ** live acc
    {
      let v1 = M.gpu_matrix_read m1 !i !sk;

      open Pulse.Lib.Array;
      pts_to_len acc;
      let sum0 = acc.(!i);
      let sum1 = sum0 `add` (v1 `mul` v2);
      acc.(!i) <- sum1;
      i := !i +^ 1sz;
    };
    pts_to_len acc;
    sk := !sk +^ 1sz;
  };
  pts_to_len acc;
}
