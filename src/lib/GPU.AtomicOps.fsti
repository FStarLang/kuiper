module GPU.AtomicOps

#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open GPU.Base
open GPU.Ref
open GPU.IntAliases

noextract
atomic
fn gpu_faa_u64
  (r : gpu_ref u64)
  (v : u64)
  (#v0 : erased u64)
  requires gpu ** gpu_pts_to r #1.0R v0
  ensures  gpu ** gpu_pts_to r #1.0R (FStar.UInt64.add_mod v v0)
