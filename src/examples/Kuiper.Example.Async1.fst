module Kuiper.Example.Async1

#lang-pulse

open Pulse.Lib
open Pulse.Lib.Pervasives
open Kuiper

module U64 = FStar.UInt64

inline_for_extraction noextract
fn kernel_f (r : gpu_ref u64) (#v : erased u64)
  ()
  requires gpu ** r |-> v
  ensures  gpu ** (r |-> U64.add_underspec v 1uL)
{
  let v = gpu_read r;
  gpu_write r (U64.add_underspec v 1uL);
}

inline_for_extraction noextract
let kernel (r : gpu_ref u64) (#v : erased u64)
  : kernel_desc _ _
  = { f = kernel_f r #v;
      full_post_sendable = solve;
      full_pre_sendable = solve
    } |> k11_as_k1n |> k1n_as_kmn |> kmn_as_kfull

fn galloc (x : u64)
  preserves cpu
  returns  r : gpu_ref u64
  ensures  on gpu_loc (r |-> x)
{
  let mut r = x;
  let gr = gpu_alloc0 #u64 ();
  Kuiper.Ref.gpu_memcpy_host_to_device gr r;
  gr
}

fn gread (gr : gpu_ref u64) (#v0 : erased u64)
  preserves cpu
  requires on gpu_loc (gr |-> v0)
  returns  v : u64
  ensures  on gpu_loc (gr |-> v) ** pure (v == v0)
{
  let mut r = 0uL;
  Kuiper.Ref.gpu_memcpy_device_to_host r gr;
  let v = !r;
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
  launch (kernel r1 #1uL); //typeclass resolution doesn't resolve the implicit
  launch (kernel r2 #2uL);
  launch (kernel r3 #3uL);
  launch (kernel r4 #4uL);
  launch (kernel r5 #5uL);
  launch (kernel r6 #6uL);

  sync_device ();

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
