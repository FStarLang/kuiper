module Kuiper.MatMulTile.Async.GMul
#lang-pulse

#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Math
open Pulse.Lib.Pledge

module A    = Pulse.Lib.Array
module SZ   = FStar.SizeT
module Kernel = Kuiper.MatMulTile.Kernel
module Barrier = Kuiper.MatMulTile.Barrier
module Prep = Kuiper.MatMulTile.Prep

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

ghost
fn recall_array_len
  (#t:Type0)
  (#len : nat)
  (a : gpu_array t len)
  (#v : Seq.seq t)
  requires
    gpu_pts_to_array a v
  ensures
    gpu_pts_to_array a v **
    pure (Seq.length v == len)
{
  unfold (gpu_pts_to_array a v);
  gpu_pts_to_slice_ref a 0 _;
  fold (gpu_pts_to_array a v);
}


fn g_mul_async
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32})
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (rows * columns))
  (#v1 : erased (seq u64))
  (#v2 : erased (seq u64))
  (#v3 : erased (seq u64))
  (#e : erased nat)
  requires
    cpu **
    epoch_live e **
    gpu_pts_to_array ga1 v1 **
    gpu_pts_to_array ga2 v2 **
    gpu_pts_to_array gr  v3 **
    pure (SZ.fits (rows * columns) /\ SZ.fits (rows * shared) /\ SZ.fits (shared * columns))
  ensures
    exists* e'.
      cpu **
      epoch_live e' **
      pure (e' >= e) **
      pledge0 (epoch_done e') (
        gpu_pts_to_array ga1 v1 **
        gpu_pts_to_array ga2 v2 **
        (exists* vr. gpu_pts_to_array gr vr) // no functional spec
      )
{
  open FStar.SizeT;
  recall_array_len ga1;
  recall_array_len ga2;
  recall_array_len gr;
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

  assert (pure (SZ.fits (rows * shared)));
  assert (pure (SZ.fits (shared * columns)));
  assert (pure (nblk * nthr == rows * columns));

  Prep.setup rows shared columns bdim nblk nthr ga1 ga2 gr v1 v2;

  assume (pure (rows_tile * columns_tile <= rows * columns));

  assume (pure (nblk <= max_threads)); // make sure to prove

  assert (pure (nthr <= 1024));
  assert (pure (2 * nthr <= 2048));
  let smem_sz = 2sz *^ nthr;
  launch_kernel_n_m_shmem_async #0
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


  ghost
  fn aux ()
    requires
      bigstar 0 (nblk * nthr) (fun i ->
        Kernel.kpost rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (Kernel.tid_to_idx rows shared columns bdim i))
    ensures
      gpu_pts_to_array ga1 v1 **
      gpu_pts_to_array ga2 v2 **
      (exists* vr. gpu_pts_to_array gr vr)
  {
    Prep.breakdown rows shared columns bdim nblk nthr ga1 ga2 gr v1 v2;
  };
  unfold pledge0;
  rewrite_pledge _ _ aux; // (fun _ -> Prep.breakdown rows shared columns bdim nblk nthr ga1 ga2 gr v1 v2);
  fold pledge0;
}

fn g_mul
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32})
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (rows * columns))
  (#v1 : erased (seq u64)) 
  (#v2 : erased (seq u64))
  (#v3 : erased (seq u64))
  requires
    cpu **
    gpu_pts_to_array ga1 v1 **
    gpu_pts_to_array ga2 v2 **
    gpu_pts_to_array gr  v3 **
    pure (SZ.fits (rows * columns) /\ SZ.fits (rows * shared) /\ SZ.fits (shared * columns))
  ensures
    cpu **
    gpu_pts_to_array ga1 v1 **
    gpu_pts_to_array ga2 v2 **
    (exists* vr. gpu_pts_to_array gr vr) // no functional spec
{
  recall_array_len ga1;
  recall_array_len ga2;
  recall_array_len gr;
  let e = get_epoch ();
  g_mul_async rows shared columns bdim ga1 ga2 gr #v1 #v2 #v3 #e;
  unfold pledge0;
  sync ();
  with e'. assert (epoch_done e');
  redeem_pledge emp_inames (epoch_done e') _;
  drop_ (epoch_done e');
  drop_ (epoch_live _);
}
