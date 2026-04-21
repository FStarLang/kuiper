module Kuiper.Example.OffsetMemcpy

(* Test that gpu_memcpy_host_to_device' and gpu_memcpy_device_to_host'
   correctly handle non-zero offsets. Catches the extraction bug where
   offsets were incorrectly multiplied by element size before EBufSub. *)

#lang-pulse

open Kuiper

module V = Pulse.Lib.Vec
module U64 = FStar.UInt64

fn main (_:unit)
  requires cpu
  returns _ : u64
  ensures cpu
{
  (* Source: [10, 20, 30, 40, 50, 60, 70, 80] *)
  let src = V.alloc 0uL 8sz;
  src.(0sz) <- 10uL;
  src.(1sz) <- 20uL;
  src.(2sz) <- 30uL;
  src.(3sz) <- 40uL;
  src.(4sz) <- 50uL;
  src.(5sz) <- 60uL;
  src.(6sz) <- 70uL;
  src.(7sz) <- 80uL;

  (* GPU array of 8 u64s, initialized to zeros *)
  let ga = gpu_array_alloc #u64 8sz;
  let zeros = V.alloc 0uL 8sz;
  Kuiper.Array.gpu_memcpy_host_to_device ga zeros 8sz;
  V.free zeros;

  (* h2d': copy 3 elements from src[1..4) to GPU[2..5)
     GPU becomes: [0, 0, 20, 30, 40, 0, 0, 0] *)
  Kuiper.Array.gpu_memcpy_host_to_device' ga 2sz #8 src 1sz 3sz;
  V.free src;

  (* d2h': copy 3 elements from GPU[2..5) to dst[3..6)
     dst becomes: [0, 0, 0, 20, 30, 40, 0, 0] *)
  let dst = V.alloc 0uL 8sz;
  Kuiper.Array.gpu_memcpy_device_to_host' #_ #_ #8 dst 3sz ga 2sz 3sz;
  gpu_array_free ga;

  let r0 = dst.(3sz);
  let r1 = dst.(4sz);
  let r2 = dst.(5sz);
  V.free dst;

  (* 20 + 30 + 40 = 90 *)
  U64.add_underspec r0 (U64.add_underspec r1 r2)
}
