module Kuiper.Example.BasicFloat

#lang-pulse

open Kuiper
open Pulse.Lib

inline_for_extraction noextract
fn kf (r : gpu_ref f32) (#v : erased f32)
  requires gpu ** r |-> v
  ensures  gpu ** (exists* v'. r |-> v')
{
  let v = gpu_read r;
  gpu_write r (add v one);
}


fn main (_:unit)
  requires cpu
  returns  _ : f32
  ensures  cpu
{
  let mut r : f32 = zero;
  let gr = gpu_alloc0 #f32 ();

  Kuiper.Ref.gpu_memcpy_host_to_device gr r;
  with v. assert on gpu_loc (gr |-> v);
  launch_kernel_1 (fun () -> kf gr #v);
  Kuiper.Ref.gpu_memcpy_device_to_host r gr;
  let v = !r;

  gpu_free gr;
  v
}
