module Kuiper.BasicFloat

#lang-pulse

open Kuiper
open Pulse.Lib

[@@CPrologue "__global__"]
fn kernel (r : gpu_ref f32) (#v : erased f32)
  requires gpu ** gpu_pts_to r v
  ensures  gpu ** (exists* v'. gpu_pts_to r v')
{
  let v = gpu_read r;
  gpu_write r (add v one);
}

fn main (_:unit)
  requires cpu
  returns  _ : f32
  ensures  cpu
{
  let r  = Box.alloc #f32 zero;
  let gr = gpu_alloc0 #f32 ();

  Box.to_ref_pts_to r;
  Kuiper.Ref.gpu_memcpy_host_to_device gr r;

  launch_kernel_1 (fun () -> kernel gr);

  Kuiper.Ref.gpu_memcpy_device_to_host r gr;
  Box.to_box_pts_to r;

  let v = Pulse.Lib.Box.(!r);

  gpu_free gr;
  Box.free r;
  v
}
