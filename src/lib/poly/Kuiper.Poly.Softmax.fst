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

type k_pointwise_exp_ty
  (et:Type0) {| floating et |} =
  (#lena : erased nat) ->
  (a : gpu_array et lena) ->
  (ebid : enatlt lena) ->
  stt unit
  (requires
    gpu **
    block_id lena ebid **
    gpu_pts_to_array1 a ebid)
  (ensures fun _ ->
    gpu **
    block_id lena ebid **
    gpu_pts_to_array1 a ebid)

inline_for_extraction noextract
fn k_pointwise_exp
  (#et : Type0) {| floating et |}
  (#lena : erased nat)
  (a : gpu_array et lena)
  (ebid : enatlt lena)
  requires
    gpu **
    block_id lena ebid **
    gpu_pts_to_array1 a ebid
  ensures
    gpu **
    block_id lena ebid **
    gpu_pts_to_array1 a ebid
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

type k_pointwise_div_ty
  (et:Type0) {| floating et |} =
  (#lena : erased nat) ->
  (a : gpu_array et lena) ->
  (d : et) ->
  (ebid : enatlt lena) ->
  stt unit
  (requires
    gpu **
    block_id lena ebid **
    gpu_pts_to_array1 a ebid)
  (ensures fun _ ->
    gpu **
    block_id lena ebid **
    gpu_pts_to_array1 a ebid)

inline_for_extraction noextract
fn k_pointwise_div
  (#et : Type0) {| floating et |}
  (#lena : erased nat)
  (a : gpu_array et lena)
  (d : et)
  (ebid : enatlt lena)
  requires
    gpu **
    block_id lena ebid **
    gpu_pts_to_array1 a ebid
  ensures
    gpu **
    block_id lena ebid **
    gpu_pts_to_array1 a ebid
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
fn softmax_gpu
  (#et : Type0) {| floating et |}
  (#lena : szp { lena < max_threads })
  (a : gpu_array et lena)
  requires cpu ** gpu_pts_to_array a 'va ** pure (lena > 0 /\ lena <= max_blocks)
  ensures  cpu ** (exists* v'. gpu_pts_to_array a v')
{
  (* RESTORE *)
  admit();

  // gpu_pts_to_ref a; (* recall length, automate *)
  // // FIXME: Annotating this should NOT be needed.
  // // Even more basic: eta-expanding the post makes the unslicing fail.
  // // Fix by adding a match_via binder_attribute on the bigstar?
  // (* Call exp on every element. *)
  // Array.gpu_array_slice_1_underspec a;

  // forevery_fromstar #(natlt lena)
  //   (fun bid -> gpu_pts_to_array1 a bid);

  // launch_kernel_n_blocks
  //   lena
  //   #(fun bid -> gpu_pts_to_array1 a bid)
  //   #(gpu_pts_to_array1 a)
  //   (fun ebid -> kexp #(SZ.v lena) a ebid);

  // forevery_tostar #(natlt lena)
  //   (fun i -> gpu_pts_to_array1 a i);
  // rewrite bigstar 0 lena (fun i -> gpu_pts_to_array1 a i)
  //      as bigstar 0 lena (gpu_pts_to_array1 a);

  // (* Reduce to sum. *)

  // Array.gpu_array_unslice_1_underspec a;

  // (* Compute average. Need swap space. *)
  // let a' = Array.gpu_array_alloc #et lena;
  // gpu_memcpy_device_to_device a' a lena;
  // Kuiper.HReduce.reduce kreduce lena a';
  // let avg = arr_read_1 zero (lena <: szp) (a' <: gpu_array et lena);
  // gpu_array_free a';

  // (* Divide by average *)
  // Array.gpu_array_slice_1_underspec a;
  // forevery_fromstar #(natlt lena)
  //   (fun bid -> gpu_pts_to_array1 a bid);
  // launch_kernel_n_blocks
  //   lena
  //   #(fun bid -> gpu_pts_to_array1 a bid)
  //   #(gpu_pts_to_array1 a)
  //    (fun ebid -> kdiv #(SZ.v lena) a avg ebid);
  // forevery_tostar #(natlt lena)
  //   (fun i -> gpu_pts_to_array1 a i);
  // rewrite bigstar 0 lena (fun i -> gpu_pts_to_array1 a i)
  //      as bigstar 0 lena (gpu_pts_to_array1 a);
  // Array.gpu_array_unslice_1_underspec a;

  // ()
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
