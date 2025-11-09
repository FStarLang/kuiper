module Kuiper.Example1

#lang-pulse

open Kuiper

module U64 = FStar.UInt64

inline_for_extraction noextract
fn kf (r : gpu_ref u64) (#v0 : erased u64)
  preserves gpu
  requires r |-> v0
  ensures  r |-> U64.add_mod v0 1uL
{
  let v = gpu_read r;
  gpu_write r (U64.add_mod v 1uL);
}

fn main (_:unit)
  preserves cpu
  requires emp
  returns  _ : u64
  ensures emp
{
  let mut r = 1uL;
  let gr = gpu_alloc0 #u64 ();

  Kuiper.Ref.gpu_memcpy_host_to_device gr r;
  with v. assert on gpu_loc (gr |-> v);
  launch_kernel_1 (fun () -> kf gr #v);

  Kuiper.Ref.gpu_memcpy_device_to_host r gr;

  let v = !r;

  assert (pure (v == 2uL));

  gpu_free gr;
  v
}
