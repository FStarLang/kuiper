module Kuiper.MatMulTile
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
module GMul = Kuiper.MatMulTile.Async.GMul

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
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32})
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

  let size = rows *^ columns;

  assert (pure (SZ.fits (rows * shared)));
  let ga1 = gpu_array_alloc #u64 (rows *^ shared);
  assert (pure (SZ.fits (shared * columns)));
  let ga2 = gpu_array_alloc #u64 (shared *^ columns);

  Kuiper.Array.gpu_memcpy_host_to_device ga1 a1 (rows *^ shared);
  Kuiper.Array.gpu_memcpy_host_to_device ga2 a2 (shared *^ columns);

  let gr = gpu_array_alloc #u64 size;

  (**)
  with v3. assert (gpu_pts_to_array gr v3);
  unfold (gpu_pts_to_array gr v3);
  gpu_pts_to_slice_ref gr 0 _;
  fold (gpu_pts_to_array gr v3);
  (**)

  GMul.g_mul rows shared columns bdim ga1 ga2 gr;

  let ar = Pulse.Lib.Array.alloc 0UL size;
  Kuiper.Array.gpu_memcpy_device_to_host ar gr size;

  gpu_array_free ga1;
  gpu_array_free ga2;
  gpu_array_free gr;

  ar
}
#pop-options
