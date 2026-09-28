module Kuiper.AtomicOps

#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open Pulse.Class.PtsTo
open Kuiper.Base
open Kuiper.Ref
open Kuiper.IntAliases
open Kuiper.Scalars

// fixme? No faa for signed ints, needs overflow check or wrapping
// addition

[@@FStar.Attributes.custard_extern "atomic_add_u32";
   FStar.Attributes.custard_c_header "kuiper/atomics.h"]
noextract
atomic
fn gpu_faa_u32
  (r : gpu_ref u32)
  (i : u32)
  preserves gpu
  requires r |-> 'v0
  returns  old : u32
  ensures  r |-> add i 'v0
  ensures  pure (old == reveal 'v0)

[@@FStar.Attributes.custard_extern "atomic_add_u64";
   FStar.Attributes.custard_c_header "kuiper/atomics.h"]
noextract
atomic
fn gpu_faa_u64
  (r : gpu_ref u64)
  (i : u64)
  preserves gpu
  requires r |-> 'v0
  returns  old : u64
  ensures  r |-> add i 'v0
  ensures  pure (old == reveal 'v0)

[@@FStar.Attributes.custard_extern "atomic_add_f32";
   FStar.Attributes.custard_c_header "kuiper/atomics.h"]
noextract
atomic
fn gpu_faa_f32
  (r : gpu_ref f32)
  (i : f32)
  preserves gpu
  requires r |-> 'v0
  returns  old : f32
  ensures  r |-> add i 'v0
  ensures  pure (old == reveal 'v0)

[@@FStar.Attributes.custard_extern "atomic_add_f64";
   FStar.Attributes.custard_c_header "kuiper/atomics.h"]
noextract
atomic
fn gpu_faa_f64
  (r : gpu_ref f64)
  (i : f64)
  preserves gpu
  requires r |-> 'v0
  returns  old : f64
  ensures  r |-> add i 'v0
  ensures  pure (old == reveal 'v0)
