module Kuiper.Kernel.Map

(* Simple kernel: pointwise map of a function on an array. *)

#lang-pulse

open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }

inline_for_extraction noextract
fn map_gpu
  (#et : Type0)
  (f: et -> et)
  (lena : szp{ lena <= max_blocks * max_threads})
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#s: erased (chest1 et lena))
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> chest_map f s)

inline_for_extraction noextract
fn map_host
  (#et : Type0) {| sized et |}
  (f: et -> et)
  (lena : szp{ lena <= max_blocks * max_threads})
  (a : Pulse.Lib.Vec.lvec et lena)
  (#s: erased (lseq et lena))
  preserves cpu
  requires  a |-> s
  ensures   a |-> lseq_map f s

(* Two-array elementwise map: a[i] := f a[i] b[i]. *)

let chest1_map2
  (#et : Type0)
  (#len : nat)
  (f : et -> et -> et)
  (sa sb : chest1 et len)
  : chest1 et len
  = mk1 (fun i -> f (acc1 sa i) (acc1 sb i))

inline_for_extraction noextract
fn map_gpu2
  (#et : Type0)
  (f : et -> et -> et)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (a : array1 et la)
  (b : array1 et lb)
  (#_ : squash (is_global a))
  (#_ : squash (is_global b))
  (#sa : erased (chest1 et lena))
  (#sb : erased (chest1 et lena))
  (#fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (b |-> Frac fb sb)
  requires  on gpu_loc (a |-> sa)
  ensures   on gpu_loc (a |-> chest1_map2 f sa sb)
