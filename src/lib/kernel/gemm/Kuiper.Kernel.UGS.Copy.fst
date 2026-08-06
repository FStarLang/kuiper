module Kuiper.Kernel.UGS.Copy

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }

module SZ = Kuiper.SizeT
module Math = FStar.Math.Lemmas

let copied_before (n k r c : nat) : bool =
  r * n + c < k

let lemma_div_bound (k m n : nat)
  : Lemma (requires n > 0 /\ k < m * n)
          (ensures k / n < m)
  = if m = 0 then ()
    else begin
      Math.small_div (n - 1) n;
      Math.lemma_div_plus (n - 1) (m - 1) n;
      Math.lemma_div_le k (m * n - 1) n
    end

let copied_before_last (m n r c : nat)
  : Lemma (requires n > 0 /\ r < m /\ c < n)
          (ensures copied_before n (m * n) r c == true)
  = Math.lemma_mult_le_left n r (m - 1);
    Math.distributivity_sub_right n m 1

let copy_final_forall
  (m n : nat)
  (vDst vSrc vDst0 : chest2 half m n)
  : Lemma
      (requires
        forall (r : natlt m) (c : natlt n).
          acc2 vDst r c ==
            (if copied_before n (m * n) r c
             then acc2 vSrc r c
             else acc2 vDst0 r c))
      (ensures
        forall (r : natlt m) (c : natlt n).
          acc2 vDst r c == acc2 vSrc r c)
  = introduce forall (r : natlt m) (c : natlt n).
      acc2 vDst r c == acc2 vSrc r c
    with copied_before_last m n r c

let copy_final_pointwise
  (m n : nat)
  (vDst vSrc vDst0 : chest2 half m n)
  : Lemma
      (requires
        forall (r : natlt m) (c : natlt n).
          acc2 vDst r c ==
            (if copied_before n (m * n) r c
             then acc2 vSrc r c
             else acc2 vDst0 r c))
      (ensures
        forall (i : Kuiper.Shape.abs
          (Kuiper.Shape.ICons m (Kuiper.Shape.ICons n Kuiper.Shape.INil))).
          acc vDst i == acc vSrc i)
  = copy_final_forall m n vDst vSrc vDst0;
    introduce forall (i : Kuiper.Shape.abs
      (Kuiper.Shape.ICons m (Kuiper.Shape.ICons n Kuiper.Shape.INil))).
      acc vDst i == acc vSrc i
    with (
      let (r, (c, ())) = i in
      assert (acc2 vDst r c == acc2 vSrc r c)
    )

inline_for_extraction noextract
fn copy_to_row_major
  (#m #n : szp)
  (#lSrc : layout2 m n) {| ctlayout lSrc |}
  (#_ : squash (SZ.fits (m * n)))
  (gSrc : array2 half lSrc { is_global gSrc })
  (gDst : array2 half (rm m n) { is_global gDst })
  (#fSrc : perm)
  (#vSrc #vDst0 : chest2 half m n)
  requires
    gpu **
    gSrc |-> Frac fSrc vSrc **
    gDst |-> vDst0
  ensures
    gpu **
    gSrc |-> Frac fSrc vSrc **
    gDst |-> vSrc
{
  let mut k : sz = 0sz;
  let kend = m *^ n;

  while (!k <^ kend)
    invariant
      gpu **
      gSrc |-> Frac fSrc vSrc **
      live k **
      pure (!k <= kend) **
      (exists* (vDst : chest2 half m n).
        gDst |-> vDst **
        pure (
          forall (r : natlt m) (c : natlt n).
            acc2 vDst r c ==
              (if copied_before n !k r c
               then acc2 vSrc r c
               else acc2 vDst0 r c)))
    decreases (kend - !k)
  {
    let kv = !k;
    assert pure (kv < kend);
    let ri : sz = kv /^ n;
    let ci : szlt n = kv %^ n;
    lemma_div_bound kv m n;
    let ri : szlt m = ri;
    let x = tensor_read gSrc (cidx2 ri ci);
    tensor_write gDst (cidx2 ri ci) x;
    k := kv +^ 1sz;
    dassert (!k <=^ kend);
  };

  with vDst. assert (gDst |-> vDst);
  copy_final_pointwise m n vDst vSrc vDst0;
  Kuiper.Chest.lemma_equal_intro vDst vSrc;
  Kuiper.Chest.ext vDst vSrc;
  rewrite (gDst |-> vDst) as (gDst |-> vSrc);
}
