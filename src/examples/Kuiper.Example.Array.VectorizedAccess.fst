module Kuiper.Example.Array.VectorizedAccess
#lang-pulse

open Kuiper

open FStar.Seq.Base

module SZ = Kuiper.SizeT
module V = Pulse.Lib.Vec
open Kuiper.Array.Vectorized

noextract
unfold
let global_id #nblk #nthr (bid : natlt nblk) (tid : natlt nthr) : natlt (nblk * nthr) = bid * nthr + tid

noextract
unfold
let kpre
  (size:sz)
  (a:larray float size)
  (s:seq float{ len s ==  SZ.v size })
  (nblk : natle max_blocks)
  (nthr : nat{(nthr*4 * nblk == SZ.v size)})
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop =
  pts_to_slice a
      (global_id bid tid * 4) (global_id bid tid * 4 + 4)
      (slice s (global_id bid tid * 4) (global_id bid tid * 4 + 4))

noextract
let scale_seq (v : float) (s : seq float)
  = Seq.init (length s) (fun i -> v `mul` (s @! i))

noextract
unfold
let kpost
  (v : float)
  (size:sz)
  (a:larray float size)
  (s:seq float{ len s == SZ.v size })
  (nblk : natle max_blocks)
  (nthr : nat{(nthr*4 * nblk == SZ.v size)})
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop =
  pts_to_slice a (global_id bid tid * 4) (global_id bid tid * 4 + 4)
      (scale_seq v (slice s (global_id bid tid * 4) (global_id bid tid * 4 + 4)))

inline_for_extraction noextract
fn kf
  (size:sz)
  (#s:erased (seq float) { len s == SZ.v size })
  (nblk : erased (natle max_blocks))
  (a:larray float size { aligned 16 a })
  (v : float)
  (nthr : sz{nthr*4 * nblk == SZ.v size})
  (bid : szlt nblk)
  (tid : szlt nthr)
  ()
requires
  gpu **
  kpre size a s nblk nthr bid tid
ensures
  gpu **
  kpost v size a s nblk nthr bid tid
{
  open Pulse.Lib.Array;
  let global_idx = ((bid *^ nthr +^ tid) *^ 4sz); rewrite each ((SZ.v bid * SZ.v nthr + SZ.v tid) * 4) as SZ.v global_idx;
  with s0. assert (pts_to_slice a (SZ.v global_idx) (SZ.v global_idx + 4) s0);

  let mut local = [| zero #float #_; 4sz |];
  Kuiper.Float32.lem_sizeof();

  assume pure (aligned 16 local); // FIXME local arrays do not need alignment
  array_vec_cpy local 0sz a global_idx;

  slice_to_array local;

  with s1. assert (local |-> s1);
  assert (pure (Seq.equal #float s0 s1));

  local.(0sz) <- v `mul` local.(0sz);
  local.(1sz) <- v `mul` local.(1sz);
  local.(2sz) <- v `mul` local.(2sz);
  local.(3sz) <- v `mul` local.(3sz);

  with s2. assert (local |-> s2);
  assert (pure (Seq.equal #float s2 (scale_seq v s1)));

  array_vec_cpy a global_idx local 0sz;

  with s3.
    assert (pts_to_slice a (SZ.v global_idx) (SZ.v global_idx + 4) s3);
  assert (pure (Seq.equal s3 s2));
  assert (pure (Seq.equal s3 (scale_seq v s0)));
  rewrite each SZ.v global_idx as ((SZ.v bid * SZ.v nthr + SZ.v tid) * 4);
  slice_to_array local;
}

fn hf (v : V.vec float)
  preserves
    exists* s. v |-> s ** pure (Seq.length s == 4)
  preserves cpu
{
  open Pulse.Lib.Array;
  let a = gpu_array_alloc #float 4sz;

  gpu_memcpy_host_to_device a v 4sz;

  with s. assert v |-> s;
  assert (pure (Seq.equal s (slice s 0 4)));
  let two = add #float one one;

  with s. assert on gpu_loc (pts_to a s);
  map_loc
    gpu_loc
    #(pts_to a s)
    #(pts_to_slice a 0 4 s ** is_full_slice a (Seq.length s))
    fn _ {};

  launch_kernel_1 (kf 4sz #s 1 a two 1sz 0sz 0sz);

  with s'. assert on gpu_loc (pts_to_slice a 0 4 s');
  map_loc
    gpu_loc
    #(pts_to_slice a 0 4 s' ** is_full_slice a (Seq.length s))
    #(pts_to a s')
    fn _ {};

  gpu_memcpy_device_to_host v a 4sz;

  gpu_array_free a;
}
