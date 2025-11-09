module Kuiper.Array.VectorizedAccess
#lang-pulse

open Kuiper

open FStar.Seq.Base

module SZ = Kuiper.SizeT
module V = Pulse.Lib.Vec
module T = FStar.Tactics.V2
open Kuiper.Array.Vectorized

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
  gpu_pts_to_slice a 
      (global_id bid tid * 4) (global_id bid tid * 4 + 4)
      (slice s (global_id bid tid * 4) (global_id bid tid * 4 + 4)) **
  pure (aligned 16 a /\ is_global_array a)

let scale_fun (v:float) (s:seq float) (i:nat{ i < length s }) : float =
  v `mul` (s @! i)

noextract
let scale_seq (v : float) (s : seq float)
  = Seq.init (length s) (fun i -> v `mul` (s @! i))

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
  gpu_pts_to_slice a (global_id bid tid * 4) (global_id bid tid * 4 + 4)
      (scale_seq v (slice s (global_id bid tid * 4) (global_id bid tid * 4 + 4)))

module T = FStar.Tactics
let is_gpu_loc (#[T.exact (`0)]gpu_id) (l:loc_id) = gpu_of l == gpu_id_loc gpu_id

open Kuiper.Seq.Common { seq_blit }
atomic
fn gpu_array_vec_cpy_dh_mine
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (dst_arr : array et)
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src_arr : gpu_array et src_sz)
  (src_off : SZ.t)
  (#src_slice_i : erased nat)
  (#src_slice_j : erased nat)
  (#f : perm)
  (#ss : erased (seq et))
  (#ds : erased (seq et))
  (#_ : squash (dst_off + chunk et <= Seq.length ds))
  (#_ : squash (src_slice_i <= src_off /\ src_off + chunk et <= src_slice_j))
  (#_ : squash (Seq.length ss == src_slice_j - src_slice_i))
  preserves gpu
  preserves gpu_pts_to_slice src_arr #f src_slice_i src_slice_j ss
  requires  pure (aligned' 16 src_arr src_off)
  requires  dst_arr |-> ds
  ensures   exists* s. dst_arr |-> s ** pure (s == seq_blit ds dst_off ss (src_off - src_slice_i) (chunk et))
{ gpu_array_vec_cpy_dh #et dst_arr dst_off #src_sz src_arr src_off #src_slice_i #src_slice_j #f #ss #ds; }

atomic
fn gpu_array_vec_cpy_hd_mine
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (#dst_sz : erased nat)
  (dst_arr : gpu_array et dst_sz)
  (dst_off : SZ.t)
  (#dst_slice_i : erased nat)
  (#dst_slice_j : erased nat)
  (src_arr : array et)
  (src_off : SZ.t)
  (#f : perm)
  (#ss : erased (seq et))
  (#ds : erased (seq et))
  (#_ : squash (dst_slice_i <= dst_off /\ dst_off + chunk et <= dst_slice_j))
  (#_ : squash (Seq.length ds == dst_slice_j - dst_slice_i))
  (#_ : squash (src_off + chunk et <= Seq.length ss))
  preserves gpu
  preserves src_arr |-> Frac f ss
  requires  pure (aligned' 16 dst_arr dst_off)
  requires  gpu_pts_to_slice dst_arr dst_slice_i dst_slice_j ds
  ensures   exists* s'. gpu_pts_to_slice dst_arr dst_slice_i dst_slice_j s' ** pure (s' == seq_blit ds (dst_off - dst_slice_i) ss src_off (chunk et))
{
  admit()
}

inline_for_extraction
fn op_Array_Access
        u#a (#t: Type u#a)
        (a: array t)
        (i: SZ.t)
        (#p: perm)
        (#s: Ghost.erased (Seq.seq t){SZ.v i < Seq.length s})
  requires pts_to a #p s
  returns  res : t
  ensures  pts_to a #p s **
           rewrites_to res (Seq.index s (SZ.v i))
{ admit() }

inline_for_extraction
fn op_Array_Assignment
        u#a (#t: Type u#a)
        (a: array t)
        (i: SZ.t)
        (v: t)
        (#s: Ghost.erased (Seq.seq t) {SZ.v i < Seq.length s})
  requires pts_to a s
  ensures  exists* s'. pts_to a s' ** pure (s' == Seq.upd s (SZ.v i) v)
{ admit() }

inline_for_extraction noextract
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
//  open Pulse.Lib.Array;
  // elim_on_gpu_array a;
  let global_idx = ((bid *^ nthr +^ tid) *^ 4sz); rewrite each ((SZ.v bid * SZ.v nthr + SZ.v tid) * 4) as SZ.v global_idx;
  with s0. assert (gpu_pts_to_slice a (SZ.v global_idx) (SZ.v global_idx + 4) s0);

  let mut local = [| zero #float #_; 4sz |];

  gpu_array_vec_cpy_dh_mine local 0sz a global_idx;
  
  with s1. assert (local |-> s1);
  assert (pure (Seq.equal #float s0 s1));

  local.(0sz) <- v `mul` local.(0sz);
  local.(1sz) <- v `mul` local.(1sz);
  local.(2sz) <- v `mul` local.(2sz);
  local.(3sz) <- v `mul` local.(3sz);

  with s2. assert (local |-> s2);
  assert (pure (Seq.equal #float s2 (scale_seq v s1)));

  gpu_array_vec_cpy_hd_mine a global_idx local 0sz;

  with s3.
    assert (gpu_pts_to_slice a (SZ.v global_idx) (SZ.v global_idx + 4) s3);
  assert (pure (Seq.equal s3 s2));
  assert (pure (Seq.equal s3 (scale_seq v s0)));
  rewrite each SZ.v global_idx as ((SZ.v bid * SZ.v nthr + SZ.v tid) * 4);
}

ghost
fn fuse_on_placeless (q:slprop) {| placeless q |} #l #p ()
requires on l p
requires q
ensures on l (p ** q)
{ 
  placeless_on_intro q l;
  admit();
  rewrite (on l p ** on l q) as on l (p ** q); 
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
  let two = add #float one one;
  fuse_on_placeless (pure (aligned 16 a /\ is_global_array a)) ();

  launch_kernel_1 (kf 4sz #(slice s 0 4) 1 a two 1sz 0sz 0sz);

  gpu_memcpy_device_to_host v a 4sz;

  gpu_array_free a;
}