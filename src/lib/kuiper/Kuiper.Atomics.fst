module Kuiper.Atomics

#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open Pulse.Class.PtsTo
open Kuiper.Base
open Kuiper.Ref
open Kuiper.IntAliases
open Kuiper.AtomicOps
open FStar.Tactics.Typeclasses { no_method }

inline_for_extraction
class has_atomic_add (t:Type) = {
  [@@@no_method]
  pure_op : t -> t -> t;
  atomic_add :
    (r : gpu_ref t) ->
    (i : t) ->
    (#v0 : erased t) ->
    stt_atomic t
      emp_inames
      (requires gpu ** r |-> v0)
      (ensures fun old ->
        ensures gpu ** (r |-> pure_op i v0) ** pure (old == reveal v0));
}

inline_for_extraction
instance has_atomic_add_u32 : has_atomic_add u32 = {
  pure_op = FStar.UInt32.add_mod;
  atomic_add = gpu_faa_u32;
}

inline_for_extraction
instance has_atomic_add_u64 : has_atomic_add u64 = {
  pure_op = FStar.UInt64.add_mod;
  atomic_add = gpu_faa_u64;
}

inline_for_extraction
instance has_atomic_add_f32 : has_atomic_add f32 = {
  pure_op = Kuiper.Float32.add;
  atomic_add = gpu_faa_f32;
}

inline_for_extraction
instance has_atomic_add_f64 : has_atomic_add f64 = {
  pure_op = Kuiper.Float64.add;
  atomic_add = gpu_faa_f64;
}
