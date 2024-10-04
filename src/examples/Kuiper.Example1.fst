module Kuiper.Example1

#lang-pulse

open Pulse.Lib
open Pulse.Lib.Pervasives
open Kuiper

module U64 = FStar.UInt64

[@@CPrologue "__global__"]
fn kernel (r : gpu_ref u64) (#v : erased u64)
  preserves gpu
  requires gpu_pts_to r v
  ensures  gpu_pts_to r (U64.add_mod v 1uL)
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
  let r  = Box.alloc #u64 1uL;
  let gr = gpu_alloc0 #u64 ();

  Box.to_ref_pts_to r;
  Kuiper.Ref.gpu_memcpy_host_to_device gr r;

  launch_kernel_1 (fun () -> kernel gr);

  Kuiper.Ref.gpu_memcpy_device_to_host r gr;
  Box.to_box_pts_to r;

  let v = Pulse.Lib.Box.(!r);

  assert (pure (v == 2uL));

  gpu_free gr;
  Box.free r;
  v
}
