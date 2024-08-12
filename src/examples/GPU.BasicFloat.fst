module GPU.BasicFloat

#lang-pulse

open Pulse.Lib
open Pulse.Lib.Pervasives
open GPU
module F = FStar.Float

[@@CPrologue "__global__"]
fn kernel (r : gpu_ref F.t) (#v : erased F.t)
  requires gpu ** gpu_pts_to r v
  ensures  gpu ** (exists* v'. gpu_pts_to r v')
{
  let v = gpu_read r;
  gpu_write r (F.add v F.one);
}

fn main (_:unit)
  requires cpu
  returns  _ : F.t
  ensures  cpu
{
  let r  = Box.alloc F.zero;
  let gr = gpu_alloc0 #F.t ();

  Box.to_ref_pts_to r;
  GPU.Ref.gpu_memcpy_host_to_device (Box.box_to_ref r) gr;

  launch_kernel_1 (fun () -> kernel gr #(hide F.zero));

  GPU.Ref.gpu_memcpy_device_to_host (Box.box_to_ref r) gr;
  Box.to_box_pts_to r;

  let v = Pulse.Lib.Box.(!r);

  gpu_free gr;
  Box.free r;
  v
}
