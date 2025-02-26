module Kuiper.MatMulTileF32
#lang-pulse

#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Math
open Pulse.Lib.Pledge

module SZ   = FStar.SizeT
module Kernel = Kuiper.MatMulTileF32.Kernel
module Barrier = Kuiper.MatMulTileF32.Barrier
module Prep = Kuiper.MatMulTileF32.Prep

fn g_mul_async
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32})
  (ga1 : gpu_array f32 (rows * shared))
  (ga2 : gpu_array f32 (shared * columns))
  (gr  : gpu_array f32 (rows * columns))
  (#v1 : erased (seq f32) { len v1 == rows * shared })
  (#v2 : erased (seq f32) { len v2 == shared * columns })
  (#v3 : erased (seq f32) { len v3 == rows * columns })
  (#e : erased nat)
  requires
    cpu **
    epoch_live e **
    (ga1 |-> v1) **
    (ga2 |-> v2) **
    (gr  |-> v3) **
    pure (SZ.fits (rows * columns) /\ SZ.fits (rows * shared) /\ SZ.fits (shared * columns))
  ensures
    exists* e'.
      cpu **
      epoch_live e' **
      pure (e' >= e) **
      pledge0 (epoch_done e') (
        (ga1 |-> v1) **
        (ga2 |-> v2) **
        (exists* vr. gr |-> vr) // no functional spec
      )
{
  open FStar.SizeT;
  // dassert (bdim < pow2 30);

  let rows_tile = div rows bdim;
  let columns_tile = SZ.div columns bdim;
  mul_pow2 30 30; // sigh
  assert (pure (pow2 30 * pow2 30 == pow2 60));
  assert (pure (rows_tile <= rows));
  assert (pure (columns_tile <= columns));
  assert (pure (SZ.fits (rows * columns)));
  Kuiper.Math.Silly.stupid_mul_mono rows_tile columns_tile rows columns;
  assert (pure (rows_tile * columns_tile <= rows * columns));
  assert (pure (SZ.fits (rows_tile * columns_tile)));
  let nblk = rows_tile *^ columns_tile;

  assert (pure (pow2 60 < pow2 64)); // trivial
  assert (pure (bdim * bdim <= 1024));
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
    f32
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
      (ga1 |-> v1) **
      (ga2 |-> v2) **
      (exists* vr. gr |-> vr)
  {
    Prep.breakdown rows shared columns bdim nblk nthr ga1 ga2 gr v1 v2;
  };
  rewrite_pledge _ _ aux;
}

fn g_mul
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32})
  (ga1 : gpu_array f32 (rows * shared))
  (ga2 : gpu_array f32 (shared * columns))
  (gr  : gpu_array f32 (rows * columns))
  (#v1 : erased (seq f32) { len v1 == rows * shared })
  (#v2 : erased (seq f32) { len v2 == shared * columns })
  (#v3 : erased (seq f32) { len v3 == rows * columns })
  requires
    cpu **
    (ga1 |-> v1) **
    (ga2 |-> v2) **
    (gr  |-> v3) **
    pure (SZ.fits (rows * columns) /\ SZ.fits (rows * shared) /\ SZ.fits (shared * columns))
  ensures
    cpu **
    (ga1 |-> v1) **
    (ga2 |-> v2) **
    (exists* vr. gr |-> vr) // no functional spec
{
  let e = get_epoch ();
  g_mul_async rows shared columns bdim ga1 ga2 gr #v1 #v2 #v3 #e;
  unfold pledge0;
  sync ();
  with e'. assert (epoch_done e');
  redeem_pledge emp_inames (epoch_done e') _;
  drop_ (epoch_done e');
  drop_ (epoch_live _);
}

#push-options "--z3rlimit 20"
fn main
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32})
  (a1 a2: vec f32)
  (v1: erased (seq f32) { len v1 == rows * shared })
  (v2: erased (seq f32) { len v2 == shared * columns })
  preserves
    cpu **
    (a1 |-> v1) **
    (a2 |-> v2)
  requires
    pure (SZ.fits (rows * columns) /\
          SZ.fits (rows * shared) /\
          SZ.fits (shared * columns) /\
          len v1 == rows * shared /\
          len v2 == shared * columns)
  returns  ar : vec f32
  ensures
    exists* vr.
      ar |-> vr
{
  open FStar.SizeT;
  dassert (rows %^ bdim = 0sz);
  dassert (columns %^ bdim = 0sz);
  dassert (shared %^ bdim = 0sz);

  let size = rows *^ columns;

  assert (pure (SZ.fits (rows * shared)));
  let ga1 = gpu_array_alloc #f32 (rows *^ shared);
  assert (pure (SZ.fits (shared * columns)));
  let ga2 = gpu_array_alloc #f32 (shared *^ columns);

  Kuiper.Array.gpu_memcpy_host_to_device ga1 a1 (rows *^ shared);
  Kuiper.Array.gpu_memcpy_host_to_device ga2 a2 (shared *^ columns);

  let gr = gpu_array_alloc #f32 size;

  (**)
  with v3. assert (gr |-> v3);
  gpu_pts_to_slice_ref gr 0 _;
  (**)

  g_mul rows shared columns bdim ga1 ga2 gr;

  let ar = Pulse.Lib.Vec.alloc Kuiper.Float32.zero size;
  Kuiper.Array.gpu_memcpy_device_to_host ar gr size;

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  ar
}
#pop-options
