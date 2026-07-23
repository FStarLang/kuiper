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
fn redeem1 (s: stream_t) (e e' : epoch_t s) (post : slprop)
  requires epoch_done e' ** pledge0 (epoch_done e) post ** pure (e' >= e)
  ensures  epoch_done e' ** post
{
  done_lower s e' e;
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

  let s1 = fresh_stream ();
  let s2 = fresh_stream ();
  let s3 = fresh_stream ();
  let s4 = fresh_stream ();
  let s5 = fresh_stream ();
  let s6 = fresh_stream ();
  let e1 = get_epoch s1 ();
  let e2 = get_epoch s2 ();
  let e3 = get_epoch s3 ();
  let e4 = get_epoch s4 ();
  let e5 = get_epoch s5 ();
  let e6 = get_epoch s6 ();
  launch (kernel r1 #1uL) s1; //typeclass resolution doesn't resolve the implicit
  launch (kernel r2 #2uL) s2;
  launch (kernel r3 #3uL) s3;
  launch (kernel r4 #4uL) s4;
  launch (kernel r5 #5uL) s5;
  launch (kernel r6 #6uL) s6;

  sync_device
    ()
    (stream_live s1 ** stream_live s2 ** stream_live s3 ** stream_live s4 ** stream_live s5 ** stream_live s6)
    (epoch_live e1 ** epoch_live e2 ** epoch_live e3 ** epoch_live e4 ** epoch_live e5 ** epoch_live e6)
    (epoch_done e1 ** epoch_done e2 ** epoch_done e3 ** epoch_done e4 ** epoch_done e5 ** epoch_done e6 **
      (exists* (e1': epoch_t s1) (e2': epoch_t s2) (e3': epoch_t s3) (e4': epoch_t s4) (e5': epoch_t s5) (e6': epoch_t s6).
      epoch_live e1' ** epoch_live e2' ** epoch_live e3' ** epoch_live e4' ** epoch_live e5' ** epoch_live e6' **
      pure (e1' >= e1 /\ e2' >= e2 /\ e3' >= e3 /\ e4' >= e4 /\ e5' >= e5 /\ e6' >= e6)))
    fn _ {
      sync_stream_ghost s1;
      sync_stream_ghost s2;
      sync_stream_ghost s3;
      sync_stream_ghost s4;
      sync_stream_ghost s5;
      sync_stream_ghost s6;
      ()
    };

  redeem1 s1 _ _ _;
  redeem1 s2 _ _ _;
  redeem1 s3 _ _ _;
  redeem1 s4 _ _ _;
  redeem1 s5 _ _ _;
  redeem1 s6 _ _ _;

  drop_ (epoch_done #s1 _);
  drop_ (epoch_done #s2 _);
  drop_ (epoch_done #s3 _);
  drop_ (epoch_done #s4 _);
  drop_ (epoch_done #s5 _);
  drop_ (epoch_done #s6 _);
  drop_ (epoch_live #s1 _);
  drop_ (epoch_live #s2 _);
  drop_ (epoch_live #s3 _);
  drop_ (epoch_live #s4 _);
  drop_ (epoch_live #s5 _);
  drop_ (epoch_live #s6 _);

  let v1 = gread r1; gpu_free r1;
  let v2 = gread r2; gpu_free r2;
  let v3 = gread r3; gpu_free r3;
  let v4 = gread r4; gpu_free r4;
  let v5 = gread r5; gpu_free r5;
  let v6 = gread r6; gpu_free r6;

  destroy_stream s1;
  destroy_stream s2;
  destroy_stream s3;
  destroy_stream s4;
  destroy_stream s5;
  destroy_stream s6;

  let v = v1 +^ v2 +^ v3 +^ v4 +^ v5 +^ v6;
  assert (pure (UInt64.v v == 2 + 3 + 4 + 5 + 6 + 7));
  v
}
