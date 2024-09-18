module Kuiper.MatMulTile
#lang-pulse

#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Math

module A    = Pulse.Lib.Array
module SZ   = FStar.SizeT
module Defs = Kuiper.MatMul.Defs
module Kernel = Kuiper.MatMulTile.Kernel
module Barrier = Kuiper.MatMulTile.Barrier

let lemma_nonneg_mul (x y : int)
  : Lemma (requires x >= 0 /\ y >= 0)
          (ensures x * y >= 0)
= ()

ghost
fn setup
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim < pow2 30})
  (nblk : erased sz { SZ.v nblk == (rows / bdim) * (columns / bdim) })
  (nthr : erased sz { SZ.v nthr == bdim * bdim
                     /\ SZ.v nblk * SZ.v nthr == rows * columns
                     /\ 2 * (shared / bdim) >= 0
                     })
  (* ^ 2nd and 3rd conjunct above just to help verifying this spec, sigh. *)
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (SZ.v nblk * SZ.v nthr))
  (v1: erased (seq u64) { Seq.length v1 == rows * shared })
  (v2: erased (seq u64) { Seq.length v2 == shared * columns })
  requires gpu_pts_to_array gr 's **
           gpu_pts_to_array ga1 v1 **
           gpu_pts_to_array ga2 v2
  ensures  bigstar 0 (SZ.v nblk * SZ.v nthr) (fun i ->
             Kernel.kpre rows shared columns ga1 ga2 gr #v1 #v2 (SZ.v nblk * SZ.v nthr)
               (Kernel.tid_to_idx rows shared columns bdim i))
{
  // Sharing the input matrices (splitting permissions)
  fold Defs.gpu_pts_to_matrix rows   shared  ga1 1 v1;
  fold Defs.gpu_pts_to_matrix shared columns ga2 1 v2;
  Defs.gpu_matrix_share_underspec #_ #1 rows   shared  ga1 (SZ.v nblk * SZ.v nthr) v1;
  Defs.gpu_matrix_share_underspec #_ #2 shared columns ga2 (SZ.v nblk * SZ.v nthr) v2;

  // Sharing the output matrix (splitting each cell)
  gpu_pts_to_ref gr; (* obtain length v == (nblk * nthr) *)
  gpu_array_slice_1 #4 gr;

  // Join resources into a single bigstar
  bigstar_zip #1 #2 #3 0 (SZ.v nblk * SZ.v nthr) _ _;
  bigstar_zip #3 #4 #0 0 (SZ.v nblk * SZ.v nthr) _ _;

  // Rewrite inside the bigstar
  ghost
  fn aux (i:nat{0 <= i /\ i < (SZ.v nblk * SZ.v nthr)})
    requires
      Defs.gpu_pts_to_matrix rows   shared  ga1 (SZ.v nblk * SZ.v nthr) v1 **
      Defs.gpu_pts_to_matrix shared columns ga2 (SZ.v nblk * SZ.v nthr) v2 **
      gpu_pts_to_array_slice gr i (i + 1) seq!['s `Seq.index` i]
    ensures
      Kernel.kpre rows shared columns ga1 ga2 gr #v1 #v2 (SZ.v nblk * SZ.v nthr) i
  {
    fold gpu_pts_to_array1 gr i;
    ()
  };
  let _ = calc (==) {
    (rows / bdim) * (columns / bdim) * bdim * bdim;
    == { magic() } // fixme, boring proof (we have divisibility)
    rows * columns;
  };
  bigstar_map #_ #_ #0 #(SZ.v nblk * SZ.v nthr) aux;
  lemma_nonneg_mul (SZ.v nblk) (SZ.v nthr);
  lemma_divides_exact rows bdim;
  lemma_divides_exact columns bdim;
  assert (pure (rows / bdim >= 1));
  assert (pure (columns / bdim >= 1));
  bigstar_permute #0 #0 #(SZ.v nblk * SZ.v nthr) #_ (Kernel.permute (rows/bdim) (columns/bdim) bdim);
  ghost fn rewrite_permute_to_fn (i: nat {0 <= i /\ i < (SZ.v nblk * SZ.v nthr)})
    requires Kernel.kpre rows shared columns ga1 ga2 gr #v1 #v2 (SZ.v nblk * SZ.v nthr) ((Kernel.permute (rows/bdim) (columns/bdim) bdim).f i)
    ensures  Kernel.kpre rows shared columns ga1 ga2 gr #v1 #v2 (SZ.v nblk * SZ.v nthr) (Kernel.tid_to_idx rows shared columns bdim i)
  {
    ()
  };
  bigstar_map #0 #0 #0 #(SZ.v nblk * SZ.v nthr) (fun i -> rewrite_permute_to_fn i);
  ()
}

ghost
fn breakdown
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim < pow2 30})
  (nblk : sz { SZ.v nblk == (rows / bdim) * (columns / bdim) })
  (nthr : sz { SZ.v nthr == bdim * bdim
                     /\ SZ.v nblk * SZ.v nthr == rows * columns
                     /\ 2 * (shared / bdim) >= 0
                     })
  (* ^ 2nd and 3rd conjunct above just to help verifying this spec, sigh. *)
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (nblk * nthr))
  (v1: erased (seq u64) { Seq.length v1 == rows * shared })
  (v2: erased (seq u64) { Seq.length v2 == shared * columns })
  requires
    bigstar 0 (nblk * nthr) (fun i ->
      Kernel.kpost rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (Kernel.tid_to_idx rows shared columns bdim i))
  ensures
    (exists* vr. gpu_pts_to_array gr vr) **
    gpu_pts_to_array ga1 v1 **
    gpu_pts_to_array ga2 v2
{
  lemma_nonneg_mul (SZ.v nblk) (SZ.v nthr);
  lemma_divides_exact rows bdim;
  lemma_divides_exact columns bdim;
  let _ = calc (==) {
    (rows / bdim) * (columns / bdim) * bdim * bdim;
    == { magic() } // fixme, boring proof (we have divisibility)
    rows * columns;
  };
  assert (pure (rows / bdim >= 1));
  assert (pure (columns / bdim >= 1));
  let perm = perm_inv (Kernel.permute (rows/bdim) (columns/bdim) bdim);
  bigstar_permute #0 #0 #(nblk * nthr) #_ perm;
  ghost fn rewrite_permute_to_fn (i: nat {0 <= i /\ i < (nblk * nthr)})
    requires Kernel.kpost rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (perm.f (Kernel.tid_to_idx rows shared columns bdim i))
    ensures  Kernel.kpost rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr) i
  {
    let once: (j: nat{ 0 <= j /\ j < (nblk * nthr) }) = Kernel.tid_to_idx rows shared columns bdim i;
    let once': (j: nat{ 0 <= j /\ j < (nblk * nthr) }) = perm.g i;
    assert (pure (once' == once));
    perm.proof once i;
    // f x == y <==> g y == x
    let double: (j: nat{ 0 <= j /\ j < (nblk * nthr) }) = perm.f once;
    assert (pure (double == i));
    rewrite (gpu_pts_to_array1 gr (perm.f (Kernel.tid_to_idx rows shared columns bdim i)))
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
  Defs.gpu_matrix_unshare_underspec #_ #1 rows   shared  ga1 (nblk * nthr) v1;
  Defs.gpu_matrix_unshare_underspec #_ #2 shared columns ga2 (nblk * nthr) v2;
  unfold Defs.gpu_pts_to_matrix rows   shared  ga1 1 v1;
  unfold Defs.gpu_pts_to_matrix shared columns ga2 1 v2;

  ()
}

let stupid_mul_mono (x y z w : nat)
: Lemma (requires x <= z /\ y <= w) (ensures x * y <= z * w)
=
  ()

#push-options "--retry 5" //sad
let stupid_divides (x:nat) (y:nonzero)
: Lemma (x/y <= x)
  [SMTPat (x/y)]
= ()
#pop-options

#push-options "--z3rlimit 20"
fn main
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim < pow2 30})
  (a1 a2: array u64)
  (v1: erased (seq u64) { Seq.length v1 == rows * shared })
  (v2: erased (seq u64) { Seq.length v2 == shared * columns })
  requires
    cpu **
    A.pts_to a1 v1 **
    A.pts_to a2 v2 **
    pure (SZ.fits (rows * columns) /\ SZ.fits (rows * shared) /\ SZ.fits (shared * columns))
  returns  ar: array u64
  ensures  cpu ** A.pts_to a1 v1 ** A.pts_to a2 v2 ** (exists* vr. A.pts_to ar vr)
{
  open FStar.SizeT;

  dassert (rows %^ bdim = 0sz);
  dassert (columns %^ bdim = 0sz);
  dassert (shared %^ bdim = 0sz);
  // dassert (bdim < pow2 30);

  let rows_tile = div rows bdim;
  let columns_tile = SZ.div columns bdim;
  mul_pow2 30 30; // sigh
  assert (pure (pow2 30 * pow2 30 == pow2 60));
  assert (pure (rows_tile <= rows));
  assert (pure (columns_tile <= columns));
  assert (pure (SZ.fits (rows * columns)));
  stupid_mul_mono rows_tile columns_tile rows columns;
  assert (pure (rows_tile * columns_tile <= rows * columns));
  assert (pure (SZ.fits (rows_tile * columns_tile)));
  let nblk = rows_tile *^ columns_tile;

  assume (pure (bdim * bdim < pow2 60)); // trivial
  assert (pure (pow2 60 < pow2 64)); // trivial
  assert (pure (SZ.fits (bdim * bdim)));
  let nthr = bdim *^ bdim;

  let _ = calc (==) {
    nblk * nthr;
    == {}
    ((rows / bdim) * (columns / bdim)) * (bdim * bdim);
    == { magic() } // just associativity, sigh
    (rows / bdim) * (columns / bdim) * bdim * bdim;
    == { magic () } // fixme, boring proof (we have divisibility)
    rows * columns;
  };

  assert (pure (SZ.fits (nblk * nthr)));
  let size = nblk *^ nthr;
  let ar = Pulse.Lib.Array.alloc 0UL size;

  assert (pure (SZ.fits (rows * shared)));
  let ga1 = gpu_array_alloc #u64 (rows *^ shared);
  assert (pure (SZ.fits (shared * columns)));
  let ga2 = gpu_array_alloc #u64 (shared *^ columns);

  Kuiper.Array.gpu_memcpy_host_to_device ga1 a1 (rows *^ shared);
  Kuiper.Array.gpu_memcpy_host_to_device ga2 a2 (shared *^ columns);

  let gr = gpu_array_alloc #u64 size;

  assert (pure (nblk * nthr == rows * columns));

  setup rows shared columns bdim (hide nblk) (hide nthr) ga1 ga2 gr v1 v2;
 
  assume (pure (rows_tile * columns_tile <= rows * columns));

  assume (pure (nblk <= max_threads)); // make sure to prove
  assume (pure (nthr <= max_threads));

  let smem_sz = 2sz *^ nthr;
  launch_kernel_n_m_sync #0
    nblk
    nthr
    #(fun (tid: nat {0 <= tid /\ tid < (nblk * nthr)} ) ->
       Kernel.kpre rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr)
         (Kernel.tid_to_idx rows shared columns bdim tid))
    #(fun (tid: nat {0 <= tid /\ tid < (nblk * nthr)} ) ->
       Kernel.kpost rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr)
         (Kernel.tid_to_idx rows shared columns bdim tid))
    u64
    smem_sz
    #(Barrier.shared_pre nthr 0)
    #(Barrier.shared_pre nthr (2 * (shared / bdim)))
    (Barrier.block_setup_ghost nthr smem_sz)
    (fun ear etid -> Kernel.kernel rows shared columns bdim ga1 ga2 gr #v1 #v2 (hide nblk) (hide nthr) smem_sz ear etid);

  breakdown rows shared columns bdim nblk nthr ga1 ga2 gr v1 v2;

  Kuiper.Array.gpu_memcpy_device_to_host ar gr size;

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  ar
}
#pop-options
