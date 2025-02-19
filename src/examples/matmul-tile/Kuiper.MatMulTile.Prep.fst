module Kuiper.MatMulTile.Prep
#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"
#lang-pulse


open Kuiper
open Kuiper.Math


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
  (nthr : nat { nthr == bdim * bdim })
  (* ^ 2nd and 3rd conjunct above just to help verifying this spec, sigh. *)
  (ga : gpu_array u64 (rows * shared))
  (gb : gpu_array u64 (shared * columns))
  (gr : gpu_array u64 (rows * columns))
  requires
    (ga |-> 'va) **
    (gb |-> 'vb) **
    (gr |-> 'vr)
  ensures 
    bigstar 0 (nblk * nthr) (fun i ->
      Kernel.kpre rows shared columns ga gb gr #'va #'vb (nblk * nthr)
        (Kernel.tid_to_idx rows shared columns bdim i))
{
  assert (pure (nblk * nthr == rows * columns)); // silly
  gpu_pts_to_ref ga;
  gpu_pts_to_ref gb;
  gpu_pts_to_ref gr;
  // Sharing the input matrices (splitting permissions)
  fold I.gpu_pts_to_matrix rows   shared  ga 1 'va;
  fold I.gpu_pts_to_matrix shared columns gb 1 'vb;
  I.gpu_matrix_share_underspec #_ #1 rows   shared  ga (nblk * nthr) 'va;
  I.gpu_matrix_share_underspec #_ #2 shared columns gb (nblk * nthr) 'vb;

  // Sharing the output matrix (splitting each cell)
  // NB: We set the implicit to get a bigstar for 0 to nblk*nthr, isntead
  // of rows*columns. This seems to work better given the pains of NLarith.
  gpu_array_slice_1 #4 #_ #(nblk * nthr) gr;

  // Join resources into a single bigstar
  bigstar_zip #1 #2 #3 0 (nblk * nthr) _ _;
  bigstar_zip #3 #4 #0 0 (nblk * nthr) _ _;

  // Rewrite inside the bigstar
  ghost
  fn aux (i:nat{0 <= i /\ i < (nblk * nthr)})
    requires
      I.gpu_pts_to_matrix rows   shared  ga (nblk * nthr) 'va **
      I.gpu_pts_to_matrix shared columns gb (nblk * nthr) 'vb **
      gpu_pts_to_slice gr i (i + 1) seq!['vr `Seq.index` i]
    ensures
      K.kpre rows shared columns ga gb gr #'va #'vb (nblk * nthr) i
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
  lemma_nonneg_mul nblk nthr;
  lemma_divides_exact rows bdim;
  lemma_divides_exact columns bdim;
  assert (pure (rows / bdim >= 1));
  assert (pure (columns / bdim >= 1));
  bigstar_permute #0 #0 #(nblk * nthr) #_ (K.permute (rows/bdim) (columns/bdim) bdim);
  ghost fn rewrite_permute_to_fn (i: nat {0 <= i /\ i < (nblk * nthr)})
    requires K.kpre rows shared columns ga gb gr #'va #'vb (nblk * nthr) ((K.permute (rows/bdim) (columns/bdim) bdim).f i)
    ensures  K.kpre rows shared columns ga gb gr #'va #'vb (nblk * nthr) (K.tid_to_idx rows shared columns bdim i)
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
  (ga : gpu_array u64 (rows * shared))
  (gb : gpu_array u64 (shared * columns))
  (gr : gpu_array u64 (rows * columns))
  requires
    bigstar 0 (nblk * nthr) (fun i ->
      K.kpost rows shared columns ga gb gr #'va #'vb (nblk * nthr) (K.tid_to_idx rows shared columns bdim i))
  ensures
    (exists* vr. gr |-> vr) **
    (ga |-> 'va) **
    (gb |-> 'vb)
{
  assert (pure (nblk * nthr == rows * columns)); // silly

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
    requires K.kpost rows shared columns ga gb gr #'va #'vb (nblk * nthr) (perm.f (K.tid_to_idx rows shared columns bdim i))
    ensures  K.kpost rows shared columns ga gb gr #'va #'vb (nblk * nthr) i
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
  I.gpu_matrix_unshare_underspec #_ #1 rows   shared  ga (nblk * nthr) 'va;
  I.gpu_matrix_unshare_underspec #_ #2 shared columns gb (nblk * nthr) 'vb;
  unfold I.gpu_pts_to_matrix rows   shared  ga 1 'va;
  unfold I.gpu_pts_to_matrix shared columns gb 1 'vb;

  ()
}
