module Kuiper.Softmax.F32

#lang-pulse
open Kuiper
module Array = Kuiper.Array
(* ^ Why do I need this? Is it because Kuiper is a module and not a namespace? *)
module Vec = Pulse.Lib.Vec

module F = Kuiper.Float32
module SZ = FStar.SizeT

[@@CPrologue "__global__"]
fn k_pointwise_exp
  (#lena : erased nat)
  (a : gpu_array f32 lena)
  (etid : tid_t { gdim_x etid == lena /\ bdim_x etid == 1 })
  requires
    gpu **
    thread_id etid **
    gpu_pts_to_array1 a (thread_index etid)
  ensures
    gpu **
    thread_id etid **
    gpu_pts_to_array1 a (thread_index etid)
{
  let i = thread_idx_all ();
  assert (pure (i < lena));
  assert (pure (SZ.v i == thread_index etid));
  unfold gpu_pts_to_array1 a (thread_index etid);
  let x = gpu_array_read #_ #_ #i #(i+1) a i;
  let x = F.exp x;
  gpu_array_write #_ #_ #i #(i+1) a i x;
  fold gpu_pts_to_array1 a (thread_index etid);
  ()
}

[@@CPrologue "__global__"]
fn k_pointwise_div
  (#lena : erased nat)
  (a : gpu_array f32 lena)
  (d : f32)
  (etid : tid_t { gdim_x etid == lena /\ bdim_x etid == 1 })
  requires
    gpu **
    thread_id etid **
    gpu_pts_to_array1 a (thread_index etid)
  ensures
    gpu **
    thread_id etid **
    gpu_pts_to_array1 a (thread_index etid)
{
  let i = thread_idx_all ();
  assert (pure (i < lena));
  assert (pure (SZ.v i == thread_index etid));
  unfold gpu_pts_to_array1 a (thread_index etid);
  let x = gpu_array_read #_ #_ #i #(i+1) a i;
  let x = x `F.div` d;
  gpu_array_write #_ #_ #i #(i+1) a i x;
  fold gpu_pts_to_array1 a (thread_index etid);
  ()
}

(* From the CPU, read one elements from a gpu array. EXTREMELY
inefficient. We need a partial "splicing" memcpy. *)
fn arr_read_1
  (len : szp)
  (a : gpu_array f32 len)
  (i : sz { i < len })
  (#f : perm)
  requires cpu ** gpu_pts_to_array a #f 'va
  returns  x : f32
  ensures  cpu ** gpu_pts_to_array a #f 'va ** pure (Seq.length 'va > 0 /\ x == Seq.head 'va)
{
  gpu_pts_to_ref a; (* automate *)
  let ca = Pulse.Lib.Vec.alloc #f32 F.zero 1sz;
  (* FIXME: Need to give lenght of ca?!? *)
  gpu_memcpy_device_to_host' #_ #_ #1 ca 0sz a 0sz 1sz;
  let x = ca.(0sz);
  Pulse.Lib.Vec.free ca;
  x;
}

fn softmax_gpu
  (#lena : szp { lena < max_threads })
  (a : gpu_array f32 lena)
  requires cpu ** gpu_pts_to_array a 'va ** pure (lena > 0 /\ lena <= max_blocks)
  ensures  cpu ** (exists* v'. gpu_pts_to_array a v')
{
  gpu_pts_to_ref a; (* recall length, automate *)
  // FIXME: Annotating this should NOT be needed.
  // Even more basic: eta-expanding the post makes the unslicing fail.
  // Fix by adding a match_via binder_attribute on the bigstar?
  (* Call exp on every element. *)
  Array.gpu_array_slice_1_underspec a;
  launch_kernel_n #0
    lena
    #(fun tid -> gpu_pts_to_array1 a tid)
    #(gpu_pts_to_array1 a)
    (fun etid -> k_pointwise_exp #(SZ.v lena) a etid);
  Array.gpu_array_unslice_1_underspec a;

  (* Compute average. Need swap space. *)
  let a' = Array.gpu_array_alloc #f32 lena;
  gpu_memcpy_device_to_device a' a lena;
  Kuiper.HReduceF32Plus.reduce lena a';
  let avg = arr_read_1 (lena <: szp) (a' <: gpu_array f32 lena) 0sz;
  gpu_array_free a';

  (* Divide by average *)
  Array.gpu_array_slice_1_underspec a;
  launch_kernel_n #0
    lena
    #(fun tid -> gpu_pts_to_array1 a tid)
    #(gpu_pts_to_array1 a)
     (fun etid -> k_pointwise_div #(SZ.v lena) a avg etid);
  Array.gpu_array_unslice_1_underspec a;

  ()
}

fn softmax
  (#lena : szp { lena < max_threads })
  (a : Vec.lvec f32 lena)
  requires cpu ** (a |-> 'va) ** pure (lena > 0 /\ lena <= max_blocks)
  ensures  cpu ** (exists* v'. a |-> v')
{
  let ga = Array.gpu_array_alloc #f32 lena;
  Array.gpu_memcpy_host_to_device #f32 ga a lena;
  softmax_gpu ga;
  gpu_memcpy_device_to_host #f32 #_ a ga lena;
  Array.gpu_array_free ga;
  ();
}
