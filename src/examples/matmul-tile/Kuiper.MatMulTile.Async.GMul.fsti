module Kuiper.MatMulTile.Async.GMul
#lang-pulse

#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Divides
open Pulse.Lib.Pledge

inline_for_extraction let x = ()

module SZ = FStar.SizeT

inline_for_extraction
fn g_mul_async
  (rows shared columns : szp)
  (bdim : szp)
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (rows * columns))
  requires
    cpu **
    epoch_live 'e0 **
    gpu_pts_to_array ga1 'v1 **
    gpu_pts_to_array ga2 'v2 **
    gpu_pts_to_array gr  'v3 **
    pure (bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32)
  ensures
    exists* e1.
      cpu **
      epoch_live e1 **
      pure (e1 >= 'e0) **
      pledge0 (epoch_done e1) (
        gpu_pts_to_array ga1 'v1 **
        gpu_pts_to_array ga2 'v2 **
        (exists* vr. gpu_pts_to_array gr vr) // no functional spec
      )

inline_for_extraction
fn g_mul
  (rows shared columns : szp)
  (bdim : szp)
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (rows * columns))
  requires
    cpu **
    gpu_pts_to_array ga1 'v1 **
    gpu_pts_to_array ga2 'v2 **
    gpu_pts_to_array gr  'v3 **
    pure (bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32)
  ensures
    cpu **
    gpu_pts_to_array ga1 'v1 **
    gpu_pts_to_array ga2 'v2 **
    (exists* vr. gpu_pts_to_array gr vr) // no functional spec
