module Kuiper.Kernel.Arange

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }

let arange_seq
  (#et : Type0)
  (len : nat { SZ.fits len })
  (f : (i:SZ.t { SZ.v i < len }) -> et)
  : GTot (lseq et len)
  = Seq.init_ghost len (fun (i : natlt len) -> f (SZ.uint_to_t i))

inline_for_extraction noextract
fn arange_gpu
  (#et : Type0) {| sized et |}
  (len : szp { len <= max_blocks * max_threads })
  (f : (i:SZ.t { SZ.v i < len }) -> et)
  preserves cpu
  returns out : Array1.t et (l1_forward len)
  ensures
    on gpu_loc (out |-> arange_seq len f) **
    pure (Array1.is_global out)
