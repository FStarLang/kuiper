module GPU.Example1

#lang-pulse

open Pulse.Lib
open Pulse.Lib.Pervasives
open GPU

module U64 = FStar.UInt64

[@@CPrologue "__global__"]
fn kernel (r : gpu_ref U64.t) (#v : erased U64.t)
  requires gpu ** gpu_pts_to r v
  ensures  gpu ** gpu_pts_to r (U64.add_underspec v 1uL)
{
  let v = gpu_read r;
  gpu_write r (U64.add_underspec v 1uL);
}

fn main (_:unit)
  requires cpu
  returns  _ : U64.t
  ensures  cpu
{
  let r  = Box.alloc #U64.t 1uL;
  let gr = gpu_alloc0 #U64.t ();
   
  Box.to_ref_pts_to r;
  GPU.Ref.gpu_memcpy_host_to_device (Box.box_to_ref r) gr;

  launch_kernel_1 (fun () -> kernel gr #(hide 1uL));

  GPU.Ref.gpu_memcpy_device_to_host (Box.box_to_ref r) gr;
  Box.to_box_pts_to r;

  let v = Pulse.Lib.Box.(!r);
   
  assert (pure (v == 2uL));
   
  gpu_free gr;
  Box.free r;
  v
}
