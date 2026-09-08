module Kuiper.Example.Async.Chain

#lang-pulse

(* Three dependent kernel launches on one stream, with no synchronization
between them. Each launch reads the preceding launch's output. *)

open Pulse.Lib
open Pulse.Lib.Pervasives
open Kuiper
open Pulse.Lib.Pledge

module U64 = FStar.UInt64

inline_for_extraction noextract
let inc (v:u64) : u64 = U64.add_underspec v 1uL

inline_for_extraction noextract
fn kernel_f (r : gpu_ref u64) (#v : erased u64)
  ()
  requires gpu ** r |-> v
  ensures  gpu ** r |-> inc v
{
  r := inc !r;
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
  let gr = alloc0 #u64 ();
  Kuiper.Ref.memcpy_host_to_device gr r;
  gr
}

fn gread (gr : gpu_ref u64) (#v0 : erased u64)
  preserves cpu
  requires on gpu_loc (gr |-> v0)
  returns  v : u64
  ensures  on gpu_loc (gr |-> v) ** pure (v == v0)
{
  let mut r = 0uL;
  Kuiper.Ref.memcpy_device_to_host r gr;
  !r;
}

fn main (_:unit)
  requires cpu
  returns  _ : u64
  ensures  cpu
{
  let r = galloc 1uL;
  let s = fresh_stream ();
  init_epoch s ();

  (* Start from an owned resource, then consume each predecessor's pledge. *)
  launch (kernel r) s;
  launch_kernel_full (kernel r) s;
  launch_kernel_full (kernel r) s;

  (* The host synchronizes only after the complete launch chain. *)
  sync_stream s;
  redeem_pledge _ _ _;
  drop_ (epoch_done s _);
  drop_ (epoch_live s _);

  let v = gread r;
  free r;
  destroy_stream s;

  assert (pure (U64.v v == 4));
  dassert (v = 4uL);
  v
}
