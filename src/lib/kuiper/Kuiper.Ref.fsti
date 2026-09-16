module Kuiper.Ref

#lang-pulse

open Pulse.Lib.Pervasives
open Kuiper.Base
open Kuiper.Sized

module R = Pulse.Lib.Reference

// Kuiper's global and shared allocations use gpu_of and block_of visibility,
// respectively. Ordinary thread-local references do not have either tag.
let is_global_ref (#a:Type0) (r:ref a) : prop =
  R.visibility_of_ref r == gpu_of

let is_shared_ref (#a:Type0) (r:ref a) : prop =
  R.visibility_of_ref r == block_of

// CUDA atomics require a global or shared address, in addition to ownership
// at the executing GPU location. Device-local stack storage is not supported.
let is_atomic_ref (#a:Type0) (r:ref a) : prop =
  is_global_ref r \/ is_shared_ref r

noextract
fn alloc0
  (#a:Type u#0)
  {| sized a |}
  ()
  preserves cpu
  requires emp
  returns x : ref a
  ensures exists* (v:a). on gpu_loc (x |-> v)
  ensures pure (R.visibility_of_ref x == gpu_of /\ R.is_full_ref x)

noextract
fn free
  (#a:Type u#0)
  (r : ref a)
  preserves cpu
  requires pure (R.is_full_ref r)
  requires on gpu_loc (r |-> ('v <: a))
  ensures emp

noextract
fn memcpy_host_to_device
  (#a:Type u#0)
  {| sized a |}
  (dst_gr : ref a)
  (src_r  : ref a)
  preserves cpu
  preserves src_r |-> Frac 'f ('v <: a)
  requires on gpu_loc (dst_gr |-> ('gv <: a))
  ensures on gpu_loc (dst_gr |-> 'v)

noextract
fn memcpy_device_to_host
  (#a:Type u#0)
  {| sized a |}
  (dst_r  : ref a)
  (src_gr : ref a)
  preserves cpu
  preserves on gpu_loc (src_gr |-> Frac 'f ('gv <: a))
  requires dst_r |-> ('v <: a)
  ensures dst_r |-> 'gv

noextract
fn memcpy_device_to_device
  (#a:Type u#0)
  {| sized a |}
  (dst_r src_gr : ref a)
  preserves cpu
  preserves on gpu_loc (src_gr |-> Frac 'f ('gv <: a))
  requires on gpu_loc (dst_r |-> ('v <: a))
  ensures on gpu_loc (dst_r |-> 'gv)
