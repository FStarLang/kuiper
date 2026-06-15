module Kuiper.Kernel.Gather

(* 2-D gather along dim=0, following torch.gather:
     out[i][j] = src[ idx[i][j] ][j]
   The 2-D arrays are passed as flat 1-D buffers in row-major order,
   parameterized by the total flat lengths lensrc, lenout and the
   common column count cols (must divide both). rows_src = lensrc / cols. *)

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
module U64 = FStar.UInt64
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.Seq.Common
open Kuiper.Tensor { ctlayout }

let idx_ok
  (rows_src : nat)
  (#len : nat)
  (si : lseq u64 len)
  : prop
  = forall (k : natlt len). U64.v (si @! k) < rows_src

let access_gather
  (#et : Type0)
  (cols : szp)
  (lensrc : nat { lensrc % SZ.v cols == 0 })
  (lenout : nat { lenout % SZ.v cols == 0 })
  (ss : lseq et lensrc)
  (si : lseq u64 lenout)
  (k : natlt lenout)
  : Pure et
    (requires U64.v (si @! k) < lensrc / SZ.v cols)
    (ensures fun _ -> True)
  = let rows_src = lensrc / SZ.v cols in
    FStar.Math.Lemmas.lemma_mult_le_right (SZ.v cols) (U64.v (si @! k)) (rows_src - 1);
    ss @! ((U64.v (si @! k)) * SZ.v cols + (k % SZ.v cols))

let lseq_gather_2d_dim0
  (#et : Type0)
  (cols : szp)
  (lensrc : nat { lensrc % SZ.v cols == 0 })
  (lenout : nat { lenout % SZ.v cols == 0 })
  (src : lseq et lensrc)
  (idx : lseq u64 lenout { idx_ok (lensrc / SZ.v cols) idx })
  : GTot (lseq et lenout)
  = Seq.init_ghost
      lenout
      (fun k -> access_gather cols lensrc lenout src idx k)

inline_for_extraction noextract
fn gather_2d_dim0_gpu
  (#et : Type0) {| sized et |}
  (cols : szp)
  (lensrc : szp { SZ.v lensrc % SZ.v cols == 0 })
  (lenout : szp { SZ.v lenout % SZ.v cols == 0 /\ lenout <= max_blocks * max_threads })
  (#ls : Array1.layout lensrc) {| ctlayout ls |}
  (#li : Array1.layout lenout) {| ctlayout li |}
  (#lo : Array1.layout lenout) {| ctlayout lo |}
  (src : Array1.t et ls { Array1.is_global src })
  (idx : Array1.t u64 li { Array1.is_global idx })
  (out : Array1.t et lo { Array1.is_global out })
  (#ss : erased (lseq et lensrc))
  (#si : erased (lseq u64 lenout))
  (#so : erased (lseq et lenout))
  (#fs #fi : perm)
  (_ : squash (idx_ok (SZ.v lensrc / SZ.v cols) si))
  norewrite
  preserves cpu
  preserves on gpu_loc (src |-> Frac fs ss) ** on gpu_loc (idx |-> Frac fi si)
  requires
    on gpu_loc (out |-> so)
  ensures
    on gpu_loc (out |-> (lseq_gather_2d_dim0 cols lensrc lenout ss si <: lseq et lenout))
