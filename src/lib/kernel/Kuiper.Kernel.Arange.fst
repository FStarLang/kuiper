module Kuiper.Kernel.Arange

(* Allocate a fresh array and initialize each cell from an index function:
   out[i] := f i. *)

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Map = Kuiper.Kernel.Map

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
{
  let out = Array1.alloc0 #et len (l1_forward len);
  Map.mapi_gpu len (fun (_ : et) (i : SZ.t { SZ.v i < len }) -> f i) out;
  with s'. assert on gpu_loc (out |-> s');
  assert pure (Seq.equal (reveal s') (arange_seq len f));
  out
}
