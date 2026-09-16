module Kuiper.Ref

#lang-pulse

open Pulse.Lib.Pervasives
open Kuiper.Base
open Kuiper.Sized

include Kuiper.Ref.Base
module R = Pulse.Lib.Reference

inline_for_extraction noextract
fn alloc0 (#a:Type0) {| sized a |} ()
  preserves cpu
  returns r:ref a
  ensures exists* (v:a). on gpu_loc (r |-> v)
  ensures pure (is_global_ref r /\ is_full_ref r)

inline_for_extraction noextract
fn free (#a:Type0) (r:ref a)
  preserves cpu
  requires pure (is_full_ref r)
  requires on gpu_loc (r |-> ('v <: a))
  ensures emp

inline_for_extraction noextract
fn read (#a:Type0) (r:ref a) (#f:perm) (#v:erased a)
  preserves r |-> Frac f v
  returns x:a
  ensures pure (x == reveal v)

inline_for_extraction noextract
let ( ! ) = read

inline_for_extraction noextract
fn write (#a:Type0) (r:ref a) (v:a)
  requires r |-> ('old <: a)
  ensures r |-> v

inline_for_extraction noextract
let ( := ) = write

inline_for_extraction noextract
fn memcpy_host_to_device (#a:Type0) {| sized a |}
  (dst:ref a) (src:R.ref a)
  preserves cpu
  preserves src |-> Frac 'f ('v <: a)
  requires on gpu_loc (dst |-> ('old <: a))
  ensures on gpu_loc (dst |-> 'v)

inline_for_extraction noextract
fn memcpy_device_to_host (#a:Type0) {| sized a |}
  (dst:R.ref a) (src:ref a)
  preserves cpu
  preserves on gpu_loc (src |-> Frac 'f ('v <: a))
  requires dst |-> ('old <: a)
  ensures dst |-> 'v

inline_for_extraction noextract
fn memcpy_device_to_device (#a:Type0) {| sized a |}
  (dst src:ref a)
  preserves cpu
  preserves on gpu_loc (src |-> Frac 'f ('v <: a))
  requires on gpu_loc (dst |-> ('old <: a))
  ensures on gpu_loc (dst |-> 'v)
