module Kuiper.AtomicOps

#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open Pulse.Class.PtsTo
open Kuiper.Base
open Kuiper.Ref
open Kuiper.PtsTo
open Kuiper.IntAliases

// fixme? No faa for signed ints, needs overflow check

noextract
atomic
fn gpu_faa_u32
  (r : gpu_ref u32)
  (i : u32)
  requires gpu ** (r |-> 'v0)
  returns  old : u32
  ensures  gpu ** (r |-> FStar.UInt32.(i +%^ 'v0)) ** pure (old == reveal 'v0)

noextract
atomic
fn gpu_faa_u64
  (r : gpu_ref u64)
  (i : u64)
  requires gpu ** (r |-> 'v0)
  returns  old : u64
  ensures  gpu ** (r |-> FStar.UInt64.(i +%^ 'v0)) ** pure (old == reveal 'v0)

noextract
atomic
fn gpu_faa_f32
  (r : gpu_ref f32)
  (i : f32)
  requires gpu ** (r |-> 'v0)
  returns  old : f32
  ensures  gpu ** (r |-> Kuiper.Float32.add i 'v0) ** pure (old == reveal 'v0)

noextract
atomic
fn gpu_faa_f64
  (r : gpu_ref f64)
  (i : f64)
  requires gpu ** (r |-> 'v0)
  returns  old : f64
  ensures  gpu ** (r |-> Kuiper.Float64.add i 'v0) ** pure (old == reveal 'v0)
