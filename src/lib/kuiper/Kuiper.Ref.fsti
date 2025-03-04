module Kuiper.Ref

#lang-pulse

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open Kuiper.Base
open Kuiper.Sized

val gpu_ref (a:Type u#0) : Type u#0

val gpu_pts_to
  (#a:Type u#0)
  ([@@@mkey]x:gpu_ref a)
  (#[exact (`1.0R)] f : perm)
  (v : a)
: slprop

unfold
instance has_pts_to_gpu_ref (a:Type) : has_pts_to (gpu_ref a) a = {
  pts_to = gpu_pts_to;
}

fn gpu_alloc0
  (#a:Type u#0)
  {| sized a |}
  ()
  preserves cpu
  requires emp
  returns  x : gpu_ref a
  ensures  exists* (v:a). x |-> v

// fn gpu_alloc
//   (#a:Type u#0)
//   {| sized a |}
//   (v:a)
//   requires cpu
//   returns  x : gpu_ref a
//   ensures  cpu ** gpu_pts_to x #1.0R v

fn gpu_free
  (#a:Type u#0)
  (r : gpu_ref a)
  preserves cpu
  requires r |-> 'v
  ensures emp

fn gpu_read
  (#a:Type u#0)
  (r : gpu_ref a)
  (#f : perm)
  (#v0 : erased a)
  preserves gpu ** pts_to r #f v0
  requires emp
  returns  v : a
  ensures  pure (v == reveal v0)

fn gpu_write
  (#a:Type u#0)
  (r : gpu_ref a)
  (v : a)
  requires gpu ** (r |-> 'v0)
  ensures  gpu ** (r |-> v)

fn gpu_memcpy_host_to_device
  (#a:Type u#0)
  {| sized a |}
  (dst_gr : gpu_ref a)
  (src_r  : ref a)
  preserves cpu ** (pts_to src_r #'f 'v)
  requires dst_gr |-> 'gv
  ensures  dst_gr |-> 'v

fn gpu_memcpy_device_to_host
  (#a:Type u#0)
  {| sized a |}
  (dst_r  : ref a)
  (src_gr : gpu_ref a)
  preserves cpu ** (pts_to src_gr #'f 'gv)
  requires dst_r |-> 'v
  ensures  dst_r |-> 'gv

fn gpu_memcpy_device_to_device
  (#a:Type u#0)
  {| sized a |}
  (dst_r  : gpu_ref a)
  (src_gr : gpu_ref a)
  preserves cpu ** (pts_to src_gr #'f 'gv)
  requires dst_r |-> 'v
  ensures  dst_r |-> 'gv
