module Kuiper.MatMulTileF32.GMul
#lang-pulse

#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Divides
open Pulse.Lib.Pledge

module SZ   = FStar.SizeT

inline_for_extraction
fn g_mul_async
  (rows shared columns : szp)
  (bdim : szp)
  (ga1 : gpu_array f32 (rows * shared))
  (ga2 : gpu_array f32 (shared * columns))
  (gr  : gpu_array f32 (rows * columns))
  (#v1 : erased (seq f32))
  (#v2 : erased (seq f32))
  (#v3 : erased (seq f32))
  (#e : erased nat)
  preserves
    cpu
  requires
    epoch_live e **
    (ga1 |-> v1) **
    (ga2 |-> v2) **
    (gr  |-> v3) **
    pure (bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32)
  ensures
    exists* e'.
      epoch_live e' **
      pure (e' >= e) **
      pledge0 (epoch_done e') (
        (ga1 |-> v1) **
        (ga2 |-> v2) **
        (exists* vr. gr |-> vr) // no functional spec
      )

inline_for_extraction
fn g_mul
  (rows shared columns : szp)
  (bdim : szp)
  (ga1 : gpu_array f32 (rows * shared))
  (ga2 : gpu_array f32 (shared * columns))
  (gr  : gpu_array f32 (rows * columns))
  (#v1 : erased (seq f32))
  (#v2 : erased (seq f32))
  (#v3 : erased (seq f32))
  preserves
    cpu **
    (ga1 |-> v1) **
    (ga2 |-> v2)
  requires
    (gr |-> v3) **
    pure (bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32)
  ensures
    (exists* vr. gr |-> vr) // no functional spec
