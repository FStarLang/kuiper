module GPU.Async1

#lang-pulse

open Pulse.Lib
open Pulse.Lib.Pervasives
open GPU

module U64 = FStar.UInt64

[@@CPrologue "__global__"]
fn kernel (r : gpu_ref u64) (#v : erased u64)
  requires gpu ** gpu_pts_to r v
  ensures  gpu ** gpu_pts_to r (U64.add_underspec v 1uL)
{
  let v = gpu_read r;
  gpu_write r (U64.add_underspec v 1uL);
}

fn galloc (x : u64)
  requires cpu
  returns  r : gpu_ref u64
  ensures  cpu ** gpu_pts_to r x
{
  let r  = Box.alloc #u64 x;
  let gr = gpu_alloc0 #u64 ();
  Box.to_ref_pts_to r;
  GPU.Ref.gpu_memcpy_host_to_device gr (Box.box_to_ref r);
  Box.to_box_pts_to r;
  Box.free r;
  gr
}

fn gread (gr : gpu_ref u64) (#v0 : erased u64)
  requires cpu ** gpu_pts_to gr v0
  returns  v : u64
  ensures  cpu ** gpu_pts_to gr v ** pure (v == v0)
{
  let r = Box.alloc #u64 0uL;
  Box.to_ref_pts_to r;
  GPU.Ref.gpu_memcpy_device_to_host (Box.box_to_ref r) gr;
  Box.to_box_pts_to r;
  let v = Pulse.Lib.Box.(!r);
  Box.free r;
  v
}

open Pulse.Lib.Pledge

[@@allow_ambiguous]
ghost
fn redeem1 (e e' : erased nat) (post : slprop)
  requires epoch_done e' ** pledge0 (epoch_done e) post ** pure (e' >= e)
  ensures  epoch_done e' ** post
{
  done_lower e' e;
  unfold pledge0;
  redeem_pledge _ _ _;
  drop_ (epoch_done e);
}

fn main (_:unit)
  requires cpu
  returns  _ : u64
  ensures  cpu
{
  open FStar.UInt64;
  let r1 = galloc 1uL;
  let r2 = galloc 2uL;
  let r3 = galloc 3uL;
  let r4 = galloc 4uL;
  let r5 = galloc 5uL;
  let r6 = galloc 6uL;

  let _ = get_epoch ();
  launch_kernel_1_async (fun () -> kernel r1);
  launch_kernel_1_async (fun () -> kernel r2);
  launch_kernel_1_async (fun () -> kernel r3);
  launch_kernel_1_async (fun () -> kernel r4);
  launch_kernel_1_async (fun () -> kernel r5);
  launch_kernel_1_async (fun () -> kernel r6);

  sync();

  redeem1 _ _ _;
  redeem1 _ _ _;
  redeem1 _ _ _;
  redeem1 _ _ _;
  redeem1 _ _ _;
  redeem1 _ _ _;

  drop_ (epoch_done _);
  drop_ (epoch_live _);

  let v1 = gread r1; gpu_free r1;
  let v2 = gread r2; gpu_free r2;
  let v3 = gread r3; gpu_free r3;
  let v4 = gread r4; gpu_free r4;
  let v5 = gread r5; gpu_free r5;
  let v6 = gread r6; gpu_free r6;

  let v = v1 +^ v2 +^ v3 +^ v4 +^ v5 +^ v6;
  assert (pure (UInt64.v v == 2 + 3 + 4 + 5 + 6 + 7));
  v
}
