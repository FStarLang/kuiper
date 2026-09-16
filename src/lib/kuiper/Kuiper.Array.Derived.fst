module Kuiper.Array.Derived

#lang-pulse

open Pulse.Lib.Vec
open Pulse.Lib.WithPure
open Pulse
open FStar.Tactics.V2
open FStar.Seq
open Kuiper.Base
open Kuiper.Sized
open Kuiper.SizeT
open Kuiper.Seq.Common
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Divides { (/?+) }
module SZ = FStar.SizeT

open Kuiper.Array.Core

ghost
fn gpu_pts_to_ref
  (#a:Type u#0)
  (#f : perm)
  (x : array a)
  (#v : seq a)
  preserves
    x |-> Frac f v
  ensures
    pure (Seq.length v == Pulse.Lib.Array.length x /\ SZ.fits (Pulse.Lib.Array.length x))
{
  Pulse.Lib.Array.pts_to_len x;
  ();
}


ghost
fn gpu_pts_to_ref_located
  (#a:Type u#0)
  (#f : perm)
  (x : array a)
  (#v : seq a)
  (#l : loc_id)
  preserves
    on l (x |-> Frac f v)
  ensures
    pure (Seq.length v == Pulse.Lib.Array.length x /\ SZ.fits (Pulse.Lib.Array.length x))
{
  ghost_impersonate l
    (on l (x |-> Frac f v))
    (on l (x |-> Frac f v) ** pure (Seq.length v == Pulse.Lib.Array.length x /\ SZ.fits (Pulse.Lib.Array.length x)))
    fn () {
      on_elim _;
      gpu_pts_to_ref x;
      on_intro (x |-> Frac f v);
    }
}

ghost
fn gpu_slice_split'
  (#a:Type u#0)
  (#sz:nat)
  (arr : larray a sz)
  (#[full_default ()] f : perm)
  (#s : erased (seq a))
  (i n : nat)
  (#_ : squash (0 <= n-i /\ n-i < length s))
  (m:nat)
  requires pts_to_slice arr #f i m s ** pure (n <= m)
  ensures  pts_to_slice arr #f i n (seq_take (n-i) s) ** pts_to_slice arr #f n m (seq_drop (n-i) s)
{
  assert pure (Seq.equal s (seq_take (n - i) s @+ seq_drop (n - i) s));
  rewrite pts_to_slice arr #f i m s
       as pts_to_slice arr #f i m (seq_take (n - i) s @+ seq_drop (n - i) s);
  slice_split arr #f i n m;
}

ghost
fn gpu_array_unslice_1'
  (#a:Type u#0)
  (#sz:nat)
  (arr : larray a sz)
  (#f : perm)
  requires array_exists arr
  requires
    (forall+ (i: natlt sz). exists* v. pts_to_cell arr #f i v)
  ensures  exists* v. pts_to arr #f v
{
  let ff = forevery_exists #(natlt sz) (fun i v -> pts_to_cell arr #f i v);
  let ss = Seq.init_ghost sz (fun i -> ff i);
  forevery_ext #(natlt sz) _ (fun i -> pts_to_cell arr #f i (ss @! i));
  array_unslice_1_with_exists arr;
}

// Preserve the vector API using array views of the host buffers.

inline_for_extraction noextract
fn memcpy_host_to_device'
  (#a:Type u#0)
  {| sized a |}
  (#dst_sz : erased nat)
  (dst_garr : larray a dst_sz)
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src_arr : vec a)
  (src_off : SZ.t)
  (cnt : SZ.t { dst_off + cnt <= dst_sz /\ src_off + cnt <= src_sz })
  (#f : perm)
  (#v : erased (seq a){ Seq.length v == src_sz })
  (#gv : erased (seq a){ Seq.length gv == dst_sz })
  preserves
    cpu **
    src_arr |-> Frac f (v <: seq a)
  requires
    on gpu_loc (dst_garr |-> gv)
  ensures
    exists* s'.
      on gpu_loc (dst_garr |-> s') **
      pure (s' == seq_blit gv dst_off v src_off cnt /\ Seq.length s' == reveal dst_sz)
{
  Pulse.Lib.Vec.to_array_pts_to src_arr;
  Kuiper.Array.Core.memcpy_host_to_device' dst_garr dst_off #src_sz (Pulse.Lib.Vec.vec_to_array src_arr) src_off cnt #f #v #gv;
  Pulse.Lib.Vec.to_vec_pts_to src_arr;
}

inline_for_extraction noextract
fn memcpy_host_to_device
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (dst_garr : larray a sz)
  (src_arr : vec a)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (seq a))
  (#gv : erased (seq a))
  preserves
    cpu **
    (src_arr |-> Frac f v)
  requires
    on gpu_loc (dst_garr |-> gv)
  requires
    pure (SZ.v cnt == sz /\
          (Pulse.Lib.Vec.length src_arr == sz \/ Seq.length v == reveal sz))
  ensures
    on gpu_loc (dst_garr |-> v) **
    pure (Seq.length v == reveal sz)
{
  Pulse.Lib.Vec.to_array_pts_to src_arr;
  Kuiper.Array.Core.memcpy_host_to_device dst_garr (Pulse.Lib.Vec.vec_to_array src_arr) cnt #f #v #gv;
  Pulse.Lib.Vec.to_vec_pts_to src_arr;
}

inline_for_extraction noextract
fn memcpy_device_to_host'
  (#a:Type u#0)
  {| sized a |}
  (#dst_sz : erased nat)
  (dst_arr : vec a)
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src_garr : larray a src_sz)
  (src_off : SZ.t)
  (cnt : SZ.t {
    dst_off + cnt <= dst_sz /\
          src_off + cnt <= src_sz
  })
  (#f : perm)
  (#v : erased (seq a){ Seq.length v == src_sz })
  (#gv : erased (seq a){ Seq.length gv == dst_sz })
  preserves
    cpu **
    on gpu_loc (src_garr |-> Frac f (v<:seq _))
  requires
    dst_arr |-> gv
  ensures
    exists* s'. dst_arr |-> s' **
    pure (s'==seq_blit gv dst_off v src_off cnt /\ Seq.length s' == reveal dst_sz)
{
  Pulse.Lib.Vec.to_array_pts_to dst_arr;
  Kuiper.Array.Core.memcpy_device_to_host' #a #_ #dst_sz (Pulse.Lib.Vec.vec_to_array dst_arr) dst_off #src_sz src_garr src_off cnt #f #v #gv;
  Pulse.Lib.Vec.to_vec_pts_to dst_arr;
}

inline_for_extraction noextract
fn memcpy_device_to_host
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (dst_arr : vec a)
  (src_garr : larray a sz)
  (cnt : SZ.t)
  (#f : perm)
  (#v : erased (seq a))
  (#gv : erased (seq a))
  preserves
    cpu **
    on gpu_loc (src_garr |-> Frac f gv)
  requires
    dst_arr |-> v **
    pure (
      SZ.v cnt == sz /\
      (Pulse.Lib.Vec.length dst_arr == sz \/ Seq.length v == reveal sz))
  ensures
    dst_arr |-> gv **
    pure (Seq.length gv == reveal sz)
{
  Pulse.Lib.Vec.to_array_pts_to dst_arr;
  Kuiper.Array.Core.memcpy_device_to_host (Pulse.Lib.Vec.vec_to_array dst_arr) src_garr cnt #f #v #gv;
  Pulse.Lib.Vec.to_vec_pts_to dst_arr;
}
