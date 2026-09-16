module Kuiper.Ref.Base

#lang-pulse

open Pulse.Lib.Pervasives
open Kuiper.Base

module A = Pulse.Lib.Array

// A reference is a one-element array. Its location and visibility come from
// that array; the type itself is independent of the memory space.
inline_for_extraction noextract
val ref (a:Type0) : Type0

inline_for_extraction noextract
val to_array (#a:Type0) (r:ref a) : A.larray a 1

inline_for_extraction noextract
val of_array (#a:Type0) (arr:A.larray a 1) : r:ref a{to_array r == arr}

let visibility_of_ref #a (r:ref a) = A.visibility_of_array (to_array r)
let is_full_ref #a (r:ref a) = A.is_full_array (to_array r)

let pts_to (#a:Type0) ([@@@mkey] r:ref a)
  (#[full_default ()] f:perm) (v:a) : slprop =
  A.pts_to (to_array r) #f seq![v]

[@@pulse_unfold]
instance has_pts_to_ref (a:Type0) : has_pts_to (ref a) a = { pts_to; }

instance val is_send_across_pts_to #a (r:ref a) #p v
  (vis:visibility{vis == visibility_of_ref r})
  : is_send_across vis (pts_to r #p v)

instance val is_send_pts_to #a (r:ref a) #p v : is_send (pts_to r #p v)

let is_global_ref #a (r:ref a) : prop = visibility_of_ref r == gpu_of
let is_shared_ref #a (r:ref a) : prop = visibility_of_ref r == block_of

// CUDA atomics support global and shared storage, including references into
// arrays. Owning a thread-local reference is insufficient.
let is_atomic_ref #a (r:ref a) : prop = is_global_ref r \/ is_shared_ref r
