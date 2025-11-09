module Kuiper.Polymorphism1

(* Testing basic polymorphism *)

#lang-pulse

open Kuiper

// inline_for_extraction noextract
[@@CPrologue "__device__"; "KrmlPrivate"]
fn kswap
  (#t : Type0)
  (#v1 #v2 : erased t)
  (r1 r2 : gpu_ref t)
  requires gpu ** (r1 |-> v1 ** r2 |-> v2)
  ensures  gpu ** (r1 |-> v2 ** r2 |-> v1)
{
  let v1 = gpu_read r1;
  let v2 = gpu_read r2;
  gpu_write r1 v2;
  gpu_write r2 v1;
}

inline_for_extraction noextract
let kernel
  (#t : Type0)
  (#v1 #v2 : _)
  (r1 r2 : gpu_ref t)
  : kernel_desc
      (r1 |-> v1 ** r2 |-> v2)
      (r1 |-> v2 ** r2 |-> v1)
  =
  { f = (fun () -> kswap r1 r2); } |> k11_as_k1n |> k1n_as_kmn |> kmn_as_kfull

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
  on_star_intro #gpu_loc (gr1 |-> _) (gr2 |-> _);
  launch_sync (kernel #t #'v1 #'v2 gr1 gr2);
  on_star_elim _ _;
  Kuiper.Ref.gpu_memcpy_device_to_host r1 gr1;
  Kuiper.Ref.gpu_memcpy_device_to_host r2 gr2;
  gpu_free gr1;
  gpu_free gr2;
}


[@@CPrologue "__device__"; "KrmlPrivate"]
fn kswap_U64
  (r1 r2 : gpu_ref u64)
  ()
  requires gpu ** (r1 |-> 'v1) ** (r2 |-> 'v2)
  ensures  gpu ** (r1 |-> 'v2) ** (r2 |-> 'v1)
{
  kswap r1 r2
}

[@@CPrologue "__device__"; "KrmlPrivate"]
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
