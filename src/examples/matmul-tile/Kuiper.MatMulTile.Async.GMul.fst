module Kuiper.MatMulTile.Async.GMul
#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"
#lang-pulse

module SZ = FStar.SizeT
open Kuiper
open Kuiper.Math
open Pulse.Lib.Pledge

module SZ   = FStar.SizeT
module Kernel = Kuiper.MatMulTile.Kernel
module Barrier = Kuiper.MatMulTile.Barrier
module Prep = Kuiper.MatMulTile.Prep

let stupid_mul_mono (x y z w : nat)
: Lemma (requires x <= z /\ y <= w) (ensures x * y <= z * w)
=
  ()

#push-options "--retry 5" // sad
let stupid_divides (x:nat) (y:pos)
: Lemma (x/y <= x)
  [SMTPat (x/y)]
= ()
#pop-options

ghost
fn recall_array_len
  (#t:Type0)
  (#alen : nat)
  (a : gpu_array t alen)
  (#v : Seq.seq t)
  requires
    a |-> v
  ensures
    (a |-> v) **
    pure (len v == alen /\ SZ.fits alen)
{
  gpu_pts_to_slice_ref a 0 _;
}

fn g_mul_async
  (rows shared columns : szp)
  (bdim : szp )
  (ga : gpu_array u64 (rows * shared))
  (gb : gpu_array u64 (shared * columns))
  (gr : gpu_array u64 (rows * columns))
  requires
    cpu **
    epoch_live 'e0 **
    (ga |-> 'va) **
    (gb |-> 'vb) **
    (gr |-> 'vr) **
    pure (bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32)
  ensures
    exists* e1.
      cpu **
      epoch_live e1 **
      pure (e1 >= 'e0) **
      pledge0 (epoch_done e1) (
        (ga |-> 'va) **
        (gb |-> 'vb) **
        (exists* vr. gr |-> vr) // no functional spec
      )
{
  open FStar.SizeT;
  recall_array_len ga;
  recall_array_len gb;
  recall_array_len gr;

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

  assert (pure (bdim * bdim <= 32 * 32));
  assert (pure (SZ.fits 1024));
  assert (pure (SZ.fits (bdim * bdim)));
  let nthr = bdim *^ bdim;

  let _ = calc (==) {
    nblk * nthr;
    == {}
    ((rows / bdim) * (columns / bdim)) * (bdim * bdim);
    == { magic() } // just associativity, sigh
    (rows / bdim) * (columns / bdim) * bdim * bdim;
    == { magic() } // fixme, boring proof (we have divisibility)
    rows * columns;
  };

  assert (pure (SZ.fits (nblk * nthr)));
  let size = nblk *^ nthr;

  assert (pure (SZ.fits (rows * shared)));
  assert (pure (SZ.fits (shared * columns)));
  assert (pure (nblk * nthr == rows * columns));

  Prep.setup rows shared columns bdim nblk nthr ga gb gr;

  assume (pure (rows_tile * columns_tile <= rows * columns));

  assume (pure (nblk <= max_threads)); // make sure to prove

  assert (pure (nthr <= 1024));
  assert (pure (2 * nthr <= 2048));
  let smem_sz = 2sz *^ nthr;
  launch_kernel_n_m_shmem_async #0
    nblk
    nthr
    #(fun (tid: nat {0 <= tid /\ tid < (nblk * nthr)} ) ->
       Kernel.kpre rows shared columns ga gb gr #'va #'vb (nblk * nthr)
         (Kernel.tid_to_idx rows shared columns bdim tid))
    #(fun (tid: nat {0 <= tid /\ tid < (nblk * nthr)} ) ->
       Kernel.kpost rows shared columns ga gb gr #'va #'vb (nblk * nthr)
         (Kernel.tid_to_idx rows shared columns bdim tid))
    u64
    smem_sz
    #(Barrier.shared_pre nthr 0)
    #(Barrier.shared_pre nthr (2 * (shared / bdim)))
    (Barrier.block_setup_ghost nthr smem_sz)
    (fun ear etid -> Kernel.kernel rows shared columns bdim ga gb gr #'va #'vb (hide nblk) (hide nthr) smem_sz ear etid);


  ghost
  fn aux ()
    requires
      bigstar 0 (nblk * nthr) (fun i ->
        Kernel.kpost rows shared columns ga gb gr #'va #'vb (nblk * nthr) (Kernel.tid_to_idx rows shared columns bdim i))
    ensures
      (ga |-> 'va) **
      (gb |-> 'vb) **
      (exists* vr. gr |-> vr)
  {
    Prep.breakdown rows shared columns bdim nblk nthr ga gb gr;
  };
  unfold pledge0;
  rewrite_pledge _ _ aux; // (fun _ -> Prep.breakdown rows shared columns bdim nblk nthr ga1 ga2 gr v1 v2);
  fold pledge0;
}

fn g_mul
  (rows shared columns bdim : szp)
  (ga : gpu_array u64 (rows * shared))
  (gb : gpu_array u64 (shared * columns))
  (gr : gpu_array u64 (rows * columns))
  preserves
    cpu ** (ga |-> 'va) ** (gb |-> 'vb)
  requires
    (gr |-> 'vr) **
    pure (bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32)
  ensures
    (exists* vr. gr |-> vr) // no functional spec
{
  recall_array_len ga;
  recall_array_len gb;
  recall_array_len gr;
  let e0 = get_epoch ();
  g_mul_async rows shared columns bdim ga gb gr;
  unfold pledge0;
  sync ();
  with e1. assert (epoch_done e1);
  redeem_pledge emp_inames (epoch_done e1) _;
  drop_ (epoch_done e1);
  drop_ (epoch_live _);
}
