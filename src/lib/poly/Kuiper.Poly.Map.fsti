module Kuiper.Poly.Map

(* Simple kernel: pointwise map of a function on an array. *)

#lang-pulse

open Kuiper
module Vec = Pulse.Lib.Vec
module SZ = Kuiper.SizeT
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.Seq.Common
open Kuiper.Tensor

inline_for_extraction noextract
val kmap
  (#et : Type0)
  (f: et -> et)
  (lena : szp{ lena <= max_blocks })
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (#_ : squash (Array1.is_global a))
  (#s : erased (lseq et lena))
  : kernel_desc
      (requires a |-> s)
      (ensures  a |-> lseq_map f s)
