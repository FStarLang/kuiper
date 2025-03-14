module Kuiper.Poly.Softmax

#lang-pulse
open Kuiper
module Array = Kuiper.Array
(* ^ Why do I need this? Is it because Kuiper is a module and not a namespace? *)
module Vec = Pulse.Lib.Vec
module SZ = FStar.SizeT

(* From the CPU, read one element from a gpu array. *)
inline_for_extraction noextract
fn arr_read_1
  (#et : Type0) {| sized et |}
  (init : et) // silly
  (len : erased nat)
  (a : gpu_array et len)
  (#f : perm)
  requires cpu ** gpu_pts_to_array a #f 'va ** pure (len > 0)
  returns  x : et
  ensures  cpu ** gpu_pts_to_array a #f 'va ** pure (Seq.length 'va > 0 /\ x == Seq.head 'va)
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
  (ebid : enatlt lena)
  ()
  requires
    gpu **
    gpu_pts_to_array1 a ebid **
    block_id lena ebid
  ensures
    gpu **
    gpu_pts_to_array1 a ebid **
    block_id lena ebid
{
  let i = get_bid ();
  assert (pure (i < lena));
  assert (pure (SZ.v i == ebid));
  unfold gpu_pts_to_array1 a ebid;
  let x = gpu_array_read #_ #_ #i #(i+1) a i;
  let x = exp x;
  gpu_array_write #_ #_ #i #(i+1) a i x;
  fold gpu_pts_to_array1 a ebid;
  ()
}

inline_for_extraction noextract
let kexp 
  (#et : Type0) {| floating et |}
  (lena : szp{ lena < max_blocks })
  (a : gpu_array et lena)
: kernel_desc
    (exists* s. gpu_pts_to_array a #1.0R s)
    (exists* s. gpu_pts_to_array a #1.0R s) =
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
  (ebid : enatlt lena)
  ()
  requires
    gpu **
    gpu_pts_to_array1 a ebid **
    block_id lena ebid
  ensures
    gpu **
    gpu_pts_to_array1 a ebid **
    block_id lena ebid
{
  let i = get_bid ();
  assert (pure (i < lena));
  assert (pure (SZ.v i == ebid));
  unfold gpu_pts_to_array1 a ebid;
  let x = gpu_array_read #_ #_ #i #(i+1) a i;
  let x = x `div` d;
  gpu_array_write #_ #_ #i #(i+1) a i x;
  fold gpu_pts_to_array1 a ebid;
  ()
}

inline_for_extraction noextract
let kdiv
  (#et : Type0) {| floating et |}
  (lena : szp{ lena < max_blocks })
  (a : gpu_array et lena)
  (d : et)
: kernel_desc
    (exists* s. gpu_pts_to_array a #1.0R s)
    (exists* s. gpu_pts_to_array a #1.0R s) =
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
  requires cpu ** gpu_pts_to_array a 'va ** pure (lena > 0 /\ lena <= max_blocks)
  ensures  cpu ** (exists* v'. gpu_pts_to_array a v')
{
  gpu_pts_to_ref a; (* recall length, automate *)

  (* Pointwise exponentiation. *)
  launch_sync (kexp lena a);

  (* Compute average. Need swap space since hreduce trashes the input. *)
  let a' = Array.gpu_array_alloc #et lena;
  gpu_memcpy_device_to_device a' a lena;
  Kuiper.Poly.HReduce.reduce lena a';
  let avg = arr_read_1 zero (lena <: szp) (a' <: gpu_array et lena);
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
  requires cpu ** (a |-> 'va) ** pure (lena > 0 /\ lena <= max_blocks)
  ensures  cpu ** (exists* v'. a |-> v')
{
  let ga = Array.gpu_array_alloc #et lena;
  Array.gpu_memcpy_host_to_device #et ga a lena;
  softmax_gpu ga;
  gpu_memcpy_device_to_host #et #_ a ga lena;
  Array.gpu_array_free ga;
  ();
}
