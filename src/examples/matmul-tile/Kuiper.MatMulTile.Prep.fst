module Kuiper.MatMulTile.Prep
#lang-pulse

#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Math

module A    = Pulse.Lib.Array
module SZ   = FStar.SizeT
module Barrier = Kuiper.MatMulTile.Barrier

module P = Kuiper.MatMul.Pure
module I = Kuiper.MatMul.Impure
module K = Kuiper.MatMulTile.Kernel

let lemma_nonneg_mul (x y : int)
  : Lemma (requires x >= 0 /\ y >= 0)
          (ensures x * y >= 0)
= ()

ghost
fn setup
  (rows shared columns : pos)
  (bdim : pos { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim < pow2 30})
  (nblk : nat { nblk == (rows / bdim) * (columns / bdim) })
  (nthr : nat { nthr == bdim * bdim
                     /\ nblk * nthr == rows * columns
                     /\ 2 * (shared / bdim) >= 0
                     })
  (* ^ 2nd and 3rd conjunct above just to help verifying this spec, sigh. *)
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (nblk * nthr))
  (v1 : seq u64 { len v1 == rows * shared })
  (v2 : seq u64 { len v2 == shared * columns })
  (#s : seq u64)
  requires gpu_pts_to_array gr s **
           gpu_pts_to_array ga1 v1 **
           gpu_pts_to_array ga2 v2
  ensures  bigstar 0 (nblk * nthr) (fun i ->
             K.kpre rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr)
               (K.tid_to_idx rows shared columns bdim i))
{
  // Sharing the input matrices (splitting permissions)
  fold I.gpu_pts_to_matrix rows   shared  ga1 1 v1;
  fold I.gpu_pts_to_matrix shared columns ga2 1 v2;
  I.gpu_matrix_share_underspec #_ #1 rows   shared  ga1 (nblk * nthr) v1;
  I.gpu_matrix_share_underspec #_ #2 shared columns ga2 (nblk * nthr) v2;

  // Sharing the output matrix (splitting each cell)
  gpu_pts_to_ref gr; (* obtain length v == (nblk * nthr) *)
  gpu_array_slice_1 #4 gr;

  // Join resources into a single bigstar
  bigstar_zip #1 #2 #3 0 (nblk * nthr) _ _;
  bigstar_zip #3 #4 #0 0 (nblk * nthr) _ _;

  // Rewrite inside the bigstar
  ghost
  fn aux (i:nat{0 <= i /\ i < (nblk * nthr)})
    requires
      I.gpu_pts_to_matrix rows   shared  ga1 (nblk * nthr) v1 **
      I.gpu_pts_to_matrix shared columns ga2 (nblk * nthr) v2 **
      gpu_pts_to_array_slice gr i (i + 1) seq![s `Seq.index` i]
    ensures
      K.kpre rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr) i
  {
    fold gpu_pts_to_array1 gr i;
    ()
  };
  let _ = calc (==) {
    (rows / bdim) * (columns / bdim) * bdim * bdim;
    == { magic() } // fixme, boring proof (we have divisibility)
    rows * columns;
  };
  bigstar_map #_ #_ #0 #(nblk * nthr) aux;
  lemma_nonneg_mul (nblk) (nthr);
  lemma_divides_exact rows bdim;
  lemma_divides_exact columns bdim;
  assert (pure (rows / bdim >= 1));
  assert (pure (columns / bdim >= 1));
  bigstar_permute #0 #0 #(nblk * nthr) #_ (K.permute (rows/bdim) (columns/bdim) bdim);
  ghost fn rewrite_permute_to_fn (i: nat {0 <= i /\ i < (nblk * nthr)})
    requires K.kpre rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr) ((K.permute (rows/bdim) (columns/bdim) bdim).f i)
    ensures  K.kpre rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (K.tid_to_idx rows shared columns bdim i)
  {
    ()
  };
  bigstar_map #0 #0 #0 #(nblk * nthr) (fun i -> rewrite_permute_to_fn i);
  ()
}

ghost
fn breakdown
  (rows shared columns : pos)
  (bdim : pos { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim < pow2 30})
  (nblk : nat { nblk == (rows / bdim) * (columns / bdim) })
  (nthr : nat { nthr == bdim * bdim
                     /\ nblk * nthr == rows * columns
                     /\ 2 * (shared / bdim) >= 0
                     })
  (* ^ 2nd and 3rd conjunct above just to help verifying this spec, sigh. *)
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (nblk * nthr))
  (v1 : seq u64 { len v1 == rows * shared })
  (v2 : seq u64 { len v2 == shared * columns })
  requires
    bigstar 0 (nblk * nthr) (fun i ->
      K.kpost rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (K.tid_to_idx rows shared columns bdim i))
  ensures
    (exists* vr. gpu_pts_to_array gr vr) **
    gpu_pts_to_array ga1 v1 **
    gpu_pts_to_array ga2 v2
{
  lemma_nonneg_mul nblk nthr;
  lemma_divides_exact rows bdim;
  lemma_divides_exact columns bdim;
  let _ = calc (==) {
    (rows / bdim) * (columns / bdim) * bdim * bdim;
    == { magic() } // fixme, boring proof (we have divisibility)
    rows * columns;
  };
  assert (pure (rows / bdim >= 1));
  assert (pure (columns / bdim >= 1));
  let perm = perm_inv (K.permute (rows/bdim) (columns/bdim) bdim);
  bigstar_permute #0 #0 #(nblk * nthr) #_ perm;
  ghost fn rewrite_permute_to_fn (i: nat {0 <= i /\ i < (nblk * nthr)})
    requires K.kpost rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (perm.f (K.tid_to_idx rows shared columns bdim i))
    ensures  K.kpost rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr) i
  {
    let once: (j: nat{ 0 <= j /\ j < (nblk * nthr) }) = K.tid_to_idx rows shared columns bdim i;
    let once': (j: nat{ 0 <= j /\ j < (nblk * nthr) }) = perm.g i;
    assert (pure (once' == once));
    perm.proof once i;
    // f x == y <==> g y == x
    let double: (j: nat{ 0 <= j /\ j < (nblk * nthr) }) = perm.f once;
    assert (pure (double == i));
    rewrite (gpu_pts_to_array1 gr (perm.f (K.tid_to_idx rows shared columns bdim i)))
         as (gpu_pts_to_array1 gr i);
    ()
  };
  admit();
  bigstar_map #0 #0 #0 #(nblk * nthr) (fun i -> rewrite_permute_to_fn i);

  // Join resources into a single bigstar
  bigstar_unzip #3 #4 #0 0 (nblk * nthr) _ _;
  bigstar_unzip #1 #2 #3 0 (nblk * nthr) _ _;

  gpu_array_unslice_1_underspec #4 gr #1.0R;

  // Unsharing the input matrices (gathering permissions)
  I.gpu_matrix_unshare_underspec #_ #1 rows   shared  ga1 (nblk * nthr) v1;
  I.gpu_matrix_unshare_underspec #_ #2 shared columns ga2 (nblk * nthr) v2;
  unfold I.gpu_pts_to_matrix rows   shared  ga1 1 v1;
  unfold I.gpu_pts_to_matrix shared columns ga2 1 v2;

  ()
}
