module Kuiper.MatMulTile.Async.GMul
#lang-pulse

open Kuiper
open Pulse.Lib.Pledge

inline_for_extraction let x = ()


inline_for_extraction
fn g_mul_async
  (rows shared columns : szp)
  (bdim : szp)
  (ga : gpu_array u64 (rows * shared))
  (gb : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (rows * columns))
  requires
    cpu **
    epoch_live 'e0 **
    (ga |-> 'va) **
    (gb |-> 'vb) **
    (gr |-> 'vr) **
    pure (bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32)
  ensures
    exists* e1.
      cpu **
      epoch_live e1 **
      pure (e1 >= 'e0) **
      pledge0 (epoch_done e1) (
        (ga |-> 'va) **
        (gb |-> 'vb) **
        (exists* vr. gr |-> vr) // no functional spec
      )

inline_for_extraction
fn g_mul
  (rows shared columns bdim : szp)
  (ga : gpu_array u64 (rows * shared))
  (gb : gpu_array u64 (shared * columns))
  (gr : gpu_array u64 (rows * columns))
  preserves
    cpu ** (ga |-> 'va) ** (gb |-> 'vb)
  requires
    (gr |-> 'vr) **
    pure (bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32)
  ensures
    (exists* vr. gr |-> vr) // no functional spec
