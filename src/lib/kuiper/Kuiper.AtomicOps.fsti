module Kuiper.AtomicOps

#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open Kuiper.Base
open Kuiper.Ref
open Kuiper.IntAliases

// fixme? No faa for signed ints, needs overflow check

noextract
atomic
fn gpu_faa_u32
  (r : gpu_ref u32)
  (v : u32)
  (#v0 : erased u32)
  requires gpu ** gpu_pts_to r v0
  returns  old : u32
  ensures  gpu ** gpu_pts_to r (FStar.UInt32.add_mod v v0) ** pure (old == reveal v0)

noextract
atomic
fn gpu_faa_u64
  (r : gpu_ref u64)
  (v : u64)
  (#v0 : erased u64)
  requires gpu ** gpu_pts_to r v0
  returns  old : u64
  ensures  gpu ** gpu_pts_to r (FStar.UInt64.add_mod v v0) ** pure (old == reveal v0)

noextract
atomic
fn gpu_faa_f32
  (r : gpu_ref f32)
  (v : f32)
  (#v0 : erased f32)
  requires gpu ** gpu_pts_to r v0
  returns  old : f32
  ensures  gpu ** gpu_pts_to r (Kuiper.Float32.add v v0) ** pure (old == reveal v0)

noextract
atomic
fn gpu_faa_f64
  (r : gpu_ref f64)
  (v : f64)
  (#v0 : erased f64)
  requires gpu ** gpu_pts_to r v0
  returns  old : f64
  ensures  gpu ** gpu_pts_to r (Kuiper.Float64.add v v0) ** pure (old == reveal v0)
