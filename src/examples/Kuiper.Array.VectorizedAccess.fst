module Kuiper.Array.VectorizedAccess
#lang-pulse

open Kuiper

open FStar.Seq.Base

module SZ = FStar.SizeT
module V = Pulse.Lib.Vec

open Kuiper.Array.Vectorized
open Kuiper.VectorType

noextract
unfold
let global_id #nblk #nthr (bid : natlt nblk) (tid : natlt nthr) : natlt (nblk * nthr) = bid * nthr + tid

noextract
unfold
let kpre
  (size:sz)
  (a:gpu_array float size)
  (s:seq float{ len s ==  SZ.v size })
  (nblk : natle max_blocks)
  (nthr : nat{(nthr*4 * nblk == SZ.v size)})
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop =
  gpu_pts_to_slice a (global_id bid tid * 4) (global_id bid tid * 4 + 4) (slice s (global_id bid tid * 4) (global_id bid tid * 4 + 4))

noextract
let scale_seq (#len : nat) (s : seq float{length s == len}) (v : float)
  = Seq.seq_of_list (List.mapT (fun x -> v `mul` x) (Seq.seq_to_list s))

noextract
unfold
let kpost
  (v : float)
  (size:sz)
  (a:gpu_array float size)
  (s:seq float{ len s == SZ.v size })
  (nblk : natle max_blocks)
  (nthr : nat{(nthr*4 * nblk == SZ.v size)})
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop =
  let s_slice = slice s (global_id bid tid * 4) (global_id bid tid * 4 + 4) in
  gpu_pts_to_slice a (global_id bid tid * 4) (global_id bid tid * 4 + 4)
    (upd_seq_vec4 s_slice
      0
      (make_float4
        ((Seq.index s_slice 0) `mul` v)
        ((Seq.index s_slice 1) `mul` v)
        ((Seq.index s_slice 2) `mul` v)
        ((Seq.index s_slice 3) `mul` v)))

// #push-options "--debug SMTFail --split_queries always"
inline_for_extraction noextract
// [@@CPrologue "__device__"] // no KrmlPrivate, example
fn kf
  (size:sz)
  (#s:erased (seq float) { len s == SZ.v size })
  (nblk : erased (natle max_blocks))
  (a:gpu_array float size)
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
  let global_idx = ((bid *^ nthr +^ tid) *^ 4sz); rewrite each ((SZ.v bid * SZ.v nthr + SZ.v tid) * 4) as SZ.v global_idx;
  let fv = gpu_array_vec4_read a global_idx;
  let x = getx fv `mul` v;
  let y = gety fv `mul` v;
  let z = getz fv `mul` v;
  let w = getw fv `mul` v;

  let fv' = make_float4 x y z w;
  gpu_array_vec4_write a global_idx fv';

  rewrite each SZ.v global_idx as ((SZ.v bid * SZ.v nthr + SZ.v tid) * 4);
  ()
}

fn hf (v : V.vec float)
  preserves
    exists* s. v |-> s ** pure (Seq.length s == 4)
  preserves cpu
{
  open Pulse.Lib.Vec;
  let a = gpu_array_alloc #float 4sz;

  gpu_memcpy_host_to_device a v 4sz;

  with s. assert v |-> s;
  assert (pure (Seq.equal s (slice s 0 4)));
  assert a |-> slice s 0 4;

  let two = Float32.one `add` one;
  launch_kernel_1 (kf 4sz #(slice s 0 4) 1 a two 1sz 0sz 0sz);

  gpu_memcpy_device_to_host v a 4sz;

  gpu_array_free a;
  ()
}
