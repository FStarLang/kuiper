module Kuiper.Example.Async.Chain

#lang-pulse

(* Dependent kernel launches on a single stream, with no synchronization in
between. Each launch reads the previous one's output, which is legal because
CUDA does not begin a kernel until everything previously enqueued on the same
stream has retired. In Kuiper this shows up as the launch consuming the pledge
that the previous launch produced, at the same queue position. *)

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
  ensures  gpu ** (r |-> inc v)
{
  let v = gpu_read r;
  gpu_write r (inc v);
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

fn main (_:unit)
  requires cpu
  returns  _ : u64
  ensures  cpu
{
  let r = galloc 1uL;
  let s = fresh_stream ();
  let e = init_epoch s ();

  (* The first launch starts from a resource we own outright. *)
  launch (kernel r #1uL) s;

  (* These two consume the preceding launch's result directly. No
  sync_stream/sync_device anywhere in the chain: the pledge produced at queue
  position n is exactly what the launch taking position n requires. *)
  launch_pledged (kernel r #(inc 1uL)) s;
  launch_pledged (kernel r #(inc (inc 1uL))) s;

  (* Only now do we need the host to see the result. *)
  sync_stream s;
  done_flushed s _;
  redeem_pledge _ _ _;
  drop_ (epoch_flushed s _);
  drop_ (epoch_done s _);
  drop_ (epoch_live s _);

  let v = gread r;
  gpu_free r;
  destroy_stream s;

  assert (pure (U64.v v == 4));
  v
}
