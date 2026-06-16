module Kuiper.Ref

#lang-pulse

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open Kuiper.Base
open Kuiper.Sized

inline_for_extraction noextract
val gpu_ref (a:Type u#0) : Type u#0

// x |-> v
inline_for_extraction noextract
val gpu_pts_to
  (#a:Type u#0)
  ([@@@mkey]x:gpu_ref a)
  (#[exact (`1.0R)] f : perm)
  (v : a)
: slprop

(* gpu refs are always in gpu global memory; sendable across any v refining gpu_of *)
instance
val is_send_across_gpu_ref
  (#a:Type u#0)
  (#f:perm)
  (r:gpu_ref a)
  (vis : visibility { vis_refines vis gpu_of })
  (v:a)
: is_send_across vis (gpu_pts_to #a r #f v)

unfold
instance has_pts_to_gpu_ref (a:Type) : has_pts_to (gpu_ref a) a = {
  pts_to = gpu_pts_to;
}

inline_for_extraction noextract
fn gpu_alloc0
  (#a:Type u#0)
  {| sized a |}
  ()
  preserves cpu
  requires emp
  returns  x : gpu_ref a
  ensures  exists* (v:a). on gpu_loc (x |-> v)

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

inline_for_extraction noextract
fn gpu_write
  (#a:Type u#0)
  (r : gpu_ref a)
  (v : a)
  requires  r |-> 'v0
  ensures   r |-> v

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
