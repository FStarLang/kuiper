module Klas.Gather

#lang-pulse
open Kuiper

module SZ = Kuiper.SizeT
module U64 = FStar.UInt64
module Array1 = Kuiper.Array1
module Gather = Kuiper.Kernel.Gather
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }

(* 2D gather along dim=0, bf16 src + u64 idx, flat row-major buffers.
   lensrc = rows_src * cols, lenout = rows_out * cols. cols must divide both. *)
fn gather_bf16_u64_2d
  (cols : szp)
  (lensrc : szp { SZ.v lensrc % SZ.v cols == 0 })
  (lenout : szp { SZ.v lenout % SZ.v cols == 0 /\ lenout <= max_blocks * max_threads })
  (src : Array1.t bf16 (l1_forward lensrc) { is_global src })
  (idx : Array1.t u64 (l1_forward lenout) { is_global idx })
  (out : Array1.t bf16 (l1_forward lenout) { is_global out })
  (#ss : erased (lseq bf16 lensrc))
  (#si : erased (lseq u64 lenout))
  (#so : erased (lseq bf16 lenout))
  (#fs #fi : perm)
  (_ : squash (Gather.idx_ok (SZ.v lensrc / SZ.v cols) si))
  norewrite
  preserves cpu
  preserves on gpu_loc (src |-> Frac fs ss) ** on gpu_loc (idx |-> Frac fi si)
  requires
    on gpu_loc (out |-> so)
  ensures
    on gpu_loc (out |-> (Gather.lseq_gather_2d_dim0 cols lensrc lenout ss si <: lseq bf16 lenout))
{
  Gather.gather_2d_dim0_gpu cols lensrc lenout src idx out ()
}
