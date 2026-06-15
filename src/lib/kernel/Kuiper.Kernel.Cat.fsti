module Kuiper.Kernel.Cat

(* Concatenate two 1D arrays at dim=0: out = a `Seq.append` b. *)

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.Seq.Common
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }

let cat2_seq
  (#et : Type0)
  (#lena #lenb : nat)
  (sa : lseq et lena)
  (sb : lseq et lenb)
  : GTot (lseq et (lena + lenb))
  = Seq.init_ghost (lena + lenb) (fun i -> if i < lena then sa @! i else sb @! (i - lena))

inline_for_extraction noextract
fn cat2_gpu
  (#et : Type0) {| sized et |}
  (lena : szp)
  (lenb : szp { SZ.fits (lena + lenb) /\ lena + lenb <= max_blocks * max_threads })
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lb : Array1.layout lenb) {| ctlayout lb |}
  (a : Array1.t et la { Array1.is_global a })
  (b : Array1.t et lb { Array1.is_global b })
  (#sa : erased (lseq et lena))
  (#sb : erased (lseq et lenb))
  (#fa #fb : perm)
  preserves cpu
  preserves on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb)
  returns out : Array1.t et (l1_forward (lena + lenb))
  ensures
    on gpu_loc (out |-> (cat2_seq sa sb <: lseq et (lena + lenb))) **
    pure (Array1.is_global out)
