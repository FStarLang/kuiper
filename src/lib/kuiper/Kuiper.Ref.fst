module Kuiper.Ref

#lang-pulse

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open FStar.Tactics.Typeclasses
open Kuiper.Base
open Kuiper.Sized
open Kuiper.Array

inline_for_extraction noextract
let gpu_ref (a:Type u#0) : Type u#0 =
  x : larray a 1 { is_global_array x /\ is_full_array x }

// x |-> v
inline_for_extraction noextract
let gpu_pts_to
  (#a:Type u#0)
  ([@@@mkey]x:gpu_ref a)
  (#[exact (`1.0R)] f : perm)
  (v : a)
  : slprop
  = Pulse.Lib.Array.pts_to x #f seq![v]

ghost
fn _unfold_loc
  (#a:Type u#0)
  (x:gpu_ref a)
  (#f:perm)
  (#v:a)
  requires
    on gpu_loc (gpu_pts_to #a x #f v)
  ensures
    on gpu_loc (Pulse.Lib.Array.pts_to x #f seq![v])
{
  map_loc gpu_loc
    #(gpu_pts_to x #f v)
    #(Pulse.Lib.Array.pts_to x #f seq![v])
    fn _ {
      unfold gpu_pts_to;
    };
}

ghost
fn _fold_loc
  (#a:Type u#0)
  (x:gpu_ref a)
  (#f:perm)
  (#v : Seq.seq a { Seq.length v > 0 })
  requires
    on gpu_loc (Pulse.Lib.Array.pts_to x #f v)
  ensures
    on gpu_loc (gpu_pts_to #a x #f (v `Seq.index` 0))
{
  map_loc gpu_loc
    #(Pulse.Lib.Array.pts_to x #f v)
    #(gpu_pts_to x #f (v `Seq.index` 0))
    fn _ {
      Pulse.Lib.Array.pts_to_len x;
      assert pure (Seq.equal v (seq![v `Seq.index` 0]));
      fold gpu_pts_to x #f (v `Seq.index` 0);
    };
}

(* gpu refs are always in gpu global memory *)
instance is_send_across_gpu_ref
  (#a:Type u#0)
  (#f:perm)
  (r:gpu_ref a)
  (v:a)
  : is_send_across gpu_of (gpu_pts_to #a r #f v)
  = Kuiper.Array.Core.is_send_pts_to _ _

inline_for_extraction noextract
fn gpu_alloc0
  (#a:Type u#0)
  {| sized a |}
  ()
  preserves cpu
  requires emp
  returns  x : gpu_ref a
  ensures  exists* (v:a). on gpu_loc (x |-> v)
{
  let x = gpu_array_alloc #a 1sz;
  with v. assert on gpu_loc (x |-> v);
  _fold_loc x;
  x
}

// fn gpu_alloc
//   (#a:Type u#0)
//   {| sized a |}
//   (v:a)
//   requires cpu
//   returns  x : gpu_ref a
//   ensures  cpu ** gpu_pts_to x #1.0R v

inline_for_extraction noextract
fn gpu_free
  (#a:Type u#0)
  (r : gpu_ref a)
  preserves cpu
  requires on gpu_loc (r |-> 'v)
  ensures emp
{
  _unfold_loc r;
  gpu_array_free r
}

inline_for_extraction noextract
fn gpu_read
  (#a:Type u#0)
  (r : gpu_ref a)
  (#f : perm)
  (#v0 : erased a)
  preserves r |-> Frac f v0
  requires emp
  returns  v : a
  ensures  pure (v == reveal v0)
{
  unfold gpu_pts_to;
  let x = r.(0sz);
  fold gpu_pts_to;
  x
}

inline_for_extraction noextract
fn gpu_write
  (#a:Type u#0)
  (r : gpu_ref a)
  (v : a)
  requires  r |-> 'v0
  ensures   r |-> v
{
  unfold gpu_pts_to;
  r.(0sz) <- v;
  with s.
    assert Pulse.Lib.Array.pts_to r #1.0R s;
  assert pure (s `Seq.equal` seq![v]);
  fold gpu_pts_to r v;
}

noextract
fn gpu_memcpy_host_to_device
  (#a:Type u#0)
  {| sized a |}
  (dst_gr : gpu_ref a)
  (src_r  : ref a)
  preserves cpu
  preserves src_r |-> Frac 'f 'v
  requires  on gpu_loc (dst_gr |-> 'gv)
  ensures   on gpu_loc (dst_gr |-> 'v)
{
  admit(); // Cannot be implemented yet, needs to treat ref as vec
}

noextract
fn gpu_memcpy_device_to_host
  (#a:Type u#0)
  {| sized a |}
  (dst_r  : ref a)
  (src_gr : gpu_ref a)
  preserves cpu
  preserves on gpu_loc (src_gr |-> Frac 'f 'gv)
  requires dst_r |-> 'v
  ensures  dst_r |-> 'gv
{
  admit(); // Cannot be implemented yet, needs to treat ref as vec
}

inline_for_extraction noextract
fn gpu_memcpy_device_to_device
  (#a:Type u#0)
  {| sized a |}
  (dst_r  : gpu_ref a)
  (src_gr : gpu_ref a)
  preserves cpu
  preserves on gpu_loc (src_gr |-> Frac 'f 'gv)
  requires on gpu_loc (dst_r |-> 'v)
  ensures  on gpu_loc (dst_r |-> 'gv)
{
  _unfold_loc dst_r;
  _unfold_loc src_gr;
  Kuiper.Array.Core.gpu_memcpy_device_to_device dst_r src_gr 1sz;
  _fold_loc dst_r;
  _fold_loc src_gr;
}
