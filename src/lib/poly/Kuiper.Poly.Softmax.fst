module Kuiper.Poly.Softmax

#lang-pulse
open Kuiper
module Array = Kuiper.Array
(* ^ Why do I need this? Is it because Kuiper is a module and not a namespace? *)
module Vec = Pulse.Lib.Vec
module SZ = Kuiper.SizeT

(* From the CPU, read one element from a gpu array. *)
inline_for_extraction noextract
fn arr_read_1
  (#et : Type0) {| sized et |}
  (init : et) // silly
  (#len : erased nat)
  (a : gpu_array et len)
  (#f : perm)
  preserves cpu ** (a |-> Frac f 'va)
  requires pure (len > 0)
  returns  x : et
  ensures  pure (Seq.length 'va > 0 /\ x == Seq.head 'va)
{
  gpu_pts_to_ref a; (* automate *)
  let ca = Pulse.Lib.Vec.alloc init 1sz;
  (* FIXME: Need to give lenght of ca?!? *)
  gpu_memcpy_device_to_host' #_ #_ #1 ca 0sz a 0sz 1sz;
  let x = ca.(0sz);
  Pulse.Lib.Vec.free ca;
  x;
}

inline_for_extraction noextract
fn kf_exp
  (#et : Type0) {| floating et |}
  (#lena : erased nat)
  (a : gpu_array et lena)
  (bid : szlt lena)
  ()
  requires
    gpu **
    gpu_pts_to_array1 a bid **
    block_id lena bid
  ensures
    gpu **
    gpu_pts_to_array1 a bid **
    block_id lena bid
{
  let i = bid; rewrite each bid as i;
  assert (pure (i < lena));
  assert (pure (SZ.v i == bid));
  unfold gpu_pts_to_array1 a i;
  gpu_pts_to_slice_ref a _ _; // Needed after API change.
  let x = gpu_array_read a i;
  let x = exp x;
  gpu_array_write a i x;
  fold gpu_pts_to_array1 a i;
  rewrite each i as bid;
  ()
}

inline_for_extraction noextract
let kexp
  (#et : Type0) {| floating et |}
  (lena : szp{ lena < max_blocks })
  (a : gpu_array et lena)
: kernel_desc
    (exists* s. a |-> s)
    (exists* s. a |-> s) =
{
  nblk = lena;
  f = kf_exp a;

  teardown = magic();
  setup = magic ();
  kpre =  gpu_pts_to_array1 a #1.0R;
  kpost = gpu_pts_to_array1 a #1.0R;
  frame = emp;
} <: kernel_desc_m_1 _ _

inline_for_extraction noextract
fn kf_div
  (#et : Type0) {| floating et |}
  (#lena : erased nat)
  (a : gpu_array et lena)
  (d : et)
  (bid : szlt lena)
  ()
  requires
    gpu **
    gpu_pts_to_array1 a bid **
    block_id lena bid
  ensures
    gpu **
    gpu_pts_to_array1 a bid **
    block_id lena bid
{
  let i = bid; rewrite each bid as i;
  assert (pure (i < lena));
  assert (pure (SZ.v i == bid));
  unfold gpu_pts_to_array1 a i;
  gpu_pts_to_slice_ref a _ _; // Needed after API change.
  let x = gpu_array_read a i;
  let x = x `div` d;
  gpu_array_write a i x;
  fold gpu_pts_to_array1 a i;
  rewrite each i as bid;
  ()
}

inline_for_extraction noextract
let kdiv
  (#et : Type0) {| floating et |}
  (lena : szp{ lena < max_blocks })
  (a : gpu_array et lena)
  (d : et)
: kernel_desc
    (exists* s. a |-> s)
    (exists* s. a |-> s) =
{
  nblk = lena;
  f = kf_div a d;

  teardown = magic();
  setup = magic ();
  kpre =  gpu_pts_to_array1 a #1.0R;
  kpost = gpu_pts_to_array1 a #1.0R;
  frame = emp;
} <: kernel_desc_m_1 _ _

inline_for_extraction noextract
fn softmax_gpu
  (#et : Type0) {| floating et |}
  (#lena : szp { lena < max_threads })
  (a : gpu_array et lena)
  preserves cpu
  requires (a |-> 'va) ** pure (lena > 0 /\ lena <= max_blocks)
  ensures  (exists* v'. a |-> v')
{
  gpu_pts_to_ref a; (* recall length, automate *)

  (* Pointwise exponentiation. *)
  launch_sync (kexp lena a);

  (* Compute average. Need swap space since hreduce trashes the input. *)
  let a' = Array.gpu_array_alloc #et lena;
  gpu_memcpy_device_to_device a' a lena;
  Kuiper.Poly.HReduce.reduce lena a';
  with s.
    unfold Kuiper.IsReduction.gpu_pts_to_slice_sum a' 0 lena s;
  let avg = arr_read_1 zero a';
  gpu_array_free a';

  (* Divide by average *)
  launch_sync (kdiv lena a avg);

  ()
}

inline_for_extraction noextract
fn softmax
  (#et : Type0) {| floating et |}
  (#lena : szp { lena < max_threads })
  (a : Vec.lvec et lena)
  preserves cpu
  requires (a |-> 'va) ** pure (lena > 0 /\ lena <= max_blocks)
  ensures  (exists* v'. a |-> v')
{
  let ga = Array.gpu_array_alloc #et lena;
  Array.gpu_memcpy_host_to_device ga a lena;
  softmax_gpu ga;
  gpu_memcpy_device_to_host a ga lena;
  Array.gpu_array_free ga;
  ();
}
