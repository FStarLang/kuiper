module GPU.Polymorphism1

(* Testing basic polymorphism *)

#lang-pulse

open Pulse
open FStar.Mul
open GPU

// inline_for_extraction noextract
fn kswap
  (#t : Type0)
  (r1 r2 : gpu_ref t)
  (#v1 #v2 : erased t)
  requires gpu ** (gpu_pts_to r1 v1 ** gpu_pts_to r2 v2)
  ensures  gpu ** (gpu_pts_to r1 v2 ** gpu_pts_to r2 v1)
{
  let v1 = gpu_read r1;
  let v2 = gpu_read r2;
  gpu_write r1 v2;
  gpu_write r2 v1;
}

inline_for_extraction noextract
fn swap_via_gpu
  (#t : Type0)
  {| d : sized t |}
  (r1 r2 : ref t)
  (#v1 #v2 : erased t)
  requires cpu ** (pts_to r1 v1 ** pts_to r2 v2)
  ensures  cpu ** (pts_to r1 v2 ** pts_to r2 v1)
{
  let gr1 = gpu_alloc0 #t #{size = d.size} ();
  let gr2 = gpu_alloc0 #t #{size = d.size} ();
  GPU.Ref.gpu_memcpy_host_to_device #t #{size = d.size} r1 gr1;
  GPU.Ref.gpu_memcpy_host_to_device #t #{size = d.size} r2 gr2;
  launch_kernel_1 (fun () -> kswap #t gr1 gr2 #v1 #v2);
  GPU.Ref.gpu_memcpy_device_to_host #t #{size = d.size} r1 gr1;
  GPU.Ref.gpu_memcpy_device_to_host #t #{size = d.size} r2 gr2;
  gpu_free gr1;
  gpu_free gr2;
}

module U64 = FStar.UInt64
module F32 = GPU.Float32

[@@CPrologue "__global__"]
fn kswap_U64
  (r1 r2 : gpu_ref u64)
  (#v1 #v2 : erased _)
  requires gpu ** (gpu_pts_to r1 v1 ** gpu_pts_to r2 v2)
  ensures  gpu ** (gpu_pts_to r1 v2 ** gpu_pts_to r2 v1)
{
  kswap r1 r2 #v1 #v2
}

[@@CPrologue "__global__"]
fn kswap_F32
  (r1 r2 : gpu_ref f32)
  (#v1 #v2 : erased _)
  requires gpu ** (gpu_pts_to r1 v1 ** gpu_pts_to r2 v2)
  ensures  gpu ** (gpu_pts_to r1 v2 ** gpu_pts_to r2 v1)
{
  kswap r1 r2 #v1 #v2
}

fn swap_U64
  (r1 r2 : ref u64)
  (#v1 #v2 : erased _)
  requires cpu ** (pts_to r1 v1 ** pts_to r2 v2)
  ensures  cpu ** (pts_to r1 v2 ** pts_to r2 v1)
{
  swap_via_gpu r1 r2 #v1 #v2
}

fn swap_F32
  (r1 r2 : ref f32)
  (#v1 #v2 : erased _)
  requires cpu ** (pts_to r1 v1 ** pts_to r2 v2)
  ensures  cpu ** (pts_to r1 v2 ** pts_to r2 v1)
{
  swap_via_gpu r1 r2 #v1 #v2
}
