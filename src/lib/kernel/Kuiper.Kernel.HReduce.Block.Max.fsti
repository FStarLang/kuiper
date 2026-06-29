module Kuiper.Kernel.HReduce.Block.Max

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Math.OnlineSoftmax { seq_max }
module SZ = Kuiper.SizeT
open Kuiper.Kernel.HReduce.Max {} (* for the [approx_function_can_approximate] instance *)

(* ── reduce_batched_block_max: one block per row, tree max-reduction in shmem ─
   MAX analogue of [Kuiper.Kernel.HReduce.Block.reduce_batched_block]. Spawns one
   block per row and performs a per-row tree reduction in shared memory using
   [fmax]/[seq_max] instead of [add]/[rsum].

   Like the 1D [Kuiper.Kernel.HReduce.Max.reduce_max], max has no real-number
   unit, so each strided bucket must be non-empty: this requires [nth <= cols].
   ───────────────────────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn reduce_batched_block_max
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (rows : szp { rows <= max_blocks })
  (cols : szp)
  (nth  : szp { nth <= max_threads /\ nth <= cols /\ SZ.fits (cols + nth) })
  (#lin  : layout2 rows cols) {| ctlayout lin  |}
  (#lout : layout1 rows)                   {| ctlayout lout |}
  (x      : array2 et lin  { is_global x      })
  (output : array1 et lout { is_global output })
  (#sx   : ematrix et   rows cols)
  (vr    : ematrix real rows cols)
  (#sout : erased (chest1 et rows))
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    on gpu_loc (output |-> sout) **
    pure (sx %~ vr)
  ensures
    exists* (sout' : chest1 et rows).
      on gpu_loc (output |-> sout') **
      pure (forall (r : nat). r < SZ.v rows ==>
            (acc1 sout' r) %~ seq_max (Kuiper.Seq.Common.lseq_map pre_map_r (ematrix_row vr r)))
