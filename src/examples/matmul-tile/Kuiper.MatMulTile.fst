module Kuiper.MatMulTile
#lang-pulse

#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Math

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

  Prep.setup rows shared columns bdim nblk nthr ga1 ga2 gr v1 v2;
 
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

  Prep.breakdown rows shared columns bdim nblk nthr ga1 ga2 gr v1 v2;

  Kuiper.Array.gpu_memcpy_device_to_host ar gr size;

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  ar
}
#pop-options
