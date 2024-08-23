module GPU.Polymorphism0

(* Testing basic polymorphism *)

#lang-pulse

open GPU

inline_for_extraction noextract
[@@CPrologue "__device__"]
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

