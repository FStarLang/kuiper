module Kuiper.MatMulTile.Async
#lang-pulse

open Kuiper
open Kuiper.Math

module A    = Pulse.Lib.Array
module SZ   = FStar.SizeT
module Kernel = Kuiper.MatMulTile.Kernel
module Barrier = Kuiper.MatMulTile.Barrier
module Prep = Kuiper.MatMulTile.Prep
module GMul = Kuiper.MatMulTile.Async.GMul
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

(* Computes (a1*a2)*(a3*a4) *)
fn main
  (nn : szp)
  (bdim : szp {bdim /? nn /\ bdim <= 32})
  (a1 a2 a3 a4 : array u64)
  preserves
    cpu **
    (a1 |-> 'v1) **
    (a2 |-> 'v2) **
    (a3 |-> 'v3) **
    (a4 |-> 'v4)
  requires
    pure (bdim /? nn /\ bdim <= 32 /\ SZ.fits (nn * nn) /\
          len 'v1 == nn * nn /\ len 'v2 == nn * nn /\
          len 'v3 == nn * nn /\ len 'v4 == nn * nn)
  returns
    ar : array u64
  ensures 
    exists* vr.
      ar |-> vr // no functional spec
{
  open FStar.SizeT;
  dassert (nn %^ bdim = 0sz);

  let size = nn *^ nn;

  assert (pure (SZ.fits (nn * nn)));
  let ga1 = gpu_array_alloc #u64 size;
  let ga2 = gpu_array_alloc #u64 size;
  let ga3 = gpu_array_alloc #u64 size;
  let ga4 = gpu_array_alloc #u64 size;

  Kuiper.Array.gpu_memcpy_host_to_device ga1 a1 size;
  Kuiper.Array.gpu_memcpy_host_to_device ga2 a2 size;
  Kuiper.Array.gpu_memcpy_host_to_device ga3 a3 size;
  Kuiper.Array.gpu_memcpy_host_to_device ga4 a4 size;

  let gt1 = gpu_array_alloc #u64 size;
  let gt2 = gpu_array_alloc #u64 size;

  (**)get_epoch();

  GMul.g_mul_async nn nn nn bdim ga1 ga2 gt1;
  GMul.g_mul_async nn nn nn bdim ga3 ga4 gt2;
  sync();
  (**) redeem1 _ _ _;
  (**) redeem1 _ _ _;
  (**) drop_ (epoch_done _);
  gpu_array_free ga2;
  gpu_array_free ga3;
  gpu_array_free ga4;
  (* do not free ga1, we reuse it for the result. *)

  GMul.g_mul_async nn nn nn bdim gt1 gt2 ga1;
  sync();
  (**) redeem1 _ _ _;
  (**) drop_ (epoch_done _);
  gpu_array_free gt1;
  gpu_array_free gt2;

  let ar = Pulse.Lib.Array.alloc 0UL size;

  Kuiper.Array.gpu_memcpy_device_to_host ar ga1 size;
  gpu_array_free ga1;
  
  (**)drop_ (epoch_live _);

  ar
}
