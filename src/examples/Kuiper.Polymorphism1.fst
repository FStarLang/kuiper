module Kuiper.Polymorphism1

(* Testing basic polymorphism *)

#lang-pulse

open Kuiper

// inline_for_extraction noextract
[@@CPrologue "__global__"]
fn kswap
  (#t : Type0)
  (r1 r2 : gpu_ref t)
  requires gpu ** ((r1 |-> 'v1) ** (r2 |-> 'v2))
  ensures  gpu ** ((r1 |-> 'v2) ** (r2 |-> 'v1))
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
  requires cpu ** (r1 |-> 'v1) ** (r2 |-> 'v2)
  ensures  cpu ** (r1 |-> 'v2) ** (r2 |-> 'v1)
{
  let gr1 = gpu_alloc0 #t ();
  let gr2 = gpu_alloc0 #t ();
  Kuiper.Ref.gpu_memcpy_host_to_device gr1 r1;
  Kuiper.Ref.gpu_memcpy_host_to_device gr2 r2;
  launch_kernel_1 (fun () -> kswap gr1 gr2);
  Kuiper.Ref.gpu_memcpy_device_to_host r1 gr1;
  Kuiper.Ref.gpu_memcpy_device_to_host r2 gr2;
  gpu_free gr1;
  gpu_free gr2;
}


[@@CPrologue "__global__"]
fn kswap_U64
  (r1 r2 : gpu_ref u64)
  requires gpu ** (r1 |-> 'v1) ** (r2 |-> 'v2)
  ensures  gpu ** (r1 |-> 'v2) ** (r2 |-> 'v1)
{
  kswap r1 r2
}

[@@CPrologue "__global__"]
fn kswap_F32
  (r1 r2 : gpu_ref f32)
  requires gpu ** (r1 |-> 'v1) ** (r2 |-> 'v2)
  ensures  gpu ** (r1 |-> 'v2) ** (r2 |-> 'v1)
{
  kswap r1 r2
}

fn swap_U64
  (r1 r2 : ref u64)
  requires cpu ** (r1 |-> 'v1) ** (r2 |-> 'v2)
  ensures  cpu ** (r1 |-> 'v2) ** (r2 |-> 'v1)
{
  swap_via_gpu r1 r2
}

fn swap_F32
  (r1 r2 : ref f32)
  requires cpu ** (r1 |-> 'v1) ** (r2 |-> 'v2)
  ensures  cpu ** (r1 |-> 'v2) ** (r2 |-> 'v1)
{
  swap_via_gpu r1 r2
}
