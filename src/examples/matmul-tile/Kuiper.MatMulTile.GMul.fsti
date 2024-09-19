module Kuiper.MatMulTile.GMul
#lang-pulse

#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Divides
open Pulse.Lib.Pledge

module SZ   = FStar.SizeT

fn g_mul_async
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32})
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (rows * columns))
  (#v1 : erased (seq u64))
  (#v2 : erased (seq u64))
  (#v3 : erased (seq u64))
  (#e : erased nat)
  requires
    cpu **
    epoch_live e **
    gpu_pts_to_array ga1 v1 **
    gpu_pts_to_array ga2 v2 **
    gpu_pts_to_array gr  v3 **
    pure (SZ.fits (rows * columns) /\ SZ.fits (rows * shared) /\ SZ.fits (shared * columns))
  ensures
    exists* e'.
      cpu **
      epoch_live e' **
      pure (e' >= e) **
      pledge0 (epoch_done e') (
        gpu_pts_to_array ga1 v1 **
        gpu_pts_to_array ga2 v2 **
        (exists* vr. gpu_pts_to_array gr vr) // no functional spec
      )

fn g_mul
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim <= 32})
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (rows * columns))
  (#v1 : erased (seq u64))
  (#v2 : erased (seq u64))
  (#v3 : erased (seq u64))
  requires
    cpu **
    gpu_pts_to_array ga1 v1 **
    gpu_pts_to_array ga2 v2 **
    gpu_pts_to_array gr  v3 **
    pure (SZ.fits (rows * columns) /\ SZ.fits (rows * shared) /\ SZ.fits (shared * columns))
  ensures
    cpu **
    gpu_pts_to_array ga1 v1 **
    gpu_pts_to_array ga2 v2 **
    (exists* vr. gpu_pts_to_array gr vr) // no functional spec
