module Kuiper.Polymorphism0

(* Testing basic polymorphism *)

#lang-pulse

open Kuiper

inline_for_extraction noextract
[@@CPrologue "__device__"]
fn kswap
  (#t : Type0)
  (r1 r2 : gpu_ref t)
  requires gpu ** (r1 |-> 'v1) ** (r2 |-> 'v2)
  ensures  gpu ** (r1 |-> 'v2) ** (r2 |-> 'v1)
{
  let v1 = gpu_read r1;
  let v2 = gpu_read r2;
  gpu_write r1 v2;
  gpu_write r2 v1;
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

