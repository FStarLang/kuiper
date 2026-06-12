module Kuiper.Kernel.Map

(* Simple kernel: pointwise map of a function on an array. *)

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.Seq.Common
open Kuiper.Tensor

inline_for_extraction noextract
fn map_gpu
  (#et : Type0)
  (f: et -> et)
  (lena : szp{ lena <= max_blocks * max_threads})
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#s: erased (lseq et lena))
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> lseq_map f s)

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

let lseq_map2
  (#a #b #c : Type0)
  (#len : nat)
  (f : a -> b -> c)
  (sa : lseq a len) (sb : lseq b len)
  : GTot (lseq c len)
  = Seq.init_ghost len (fun i -> f (sa @! i) (sb @! i))

let lseq_map_cast
  (#a #b : Type0)
  (#len : nat)
  (f : a -> b)
  (sa : lseq a len)
  : GTot (lseq b len)
  = Seq.init_ghost len (fun i -> f (sa @! i))

inline_for_extraction noextract
fn map_gpu_cast
  (#et #ot : Type0)
  (f : et -> ot)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lc : Array1.layout lena) {| ctlayout lc |}
  (a : Array1.t et la)
  (c : Array1.t ot lc)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array1.is_global c))
  (#sa : erased (lseq et lena))
  (#sc : erased (lseq ot lena))
  (#fa : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa)
  requires  on gpu_loc (c |-> sc)
  ensures   on gpu_loc (c |-> (lseq_map_cast f sa <: lseq ot lena))

inline_for_extraction noextract
fn map_gpu2
  (#et : Type0)
  (f : et -> et -> et)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lb : Array1.layout lena) {| ctlayout lb |}
  (a : Array1.t et la)
  (b : Array1.t et lb)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array1.is_global b))
  (#sa : erased (lseq et lena))
  (#sb : erased (lseq et lena))
  (#fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (b |-> Frac fb sb)
  requires  on gpu_loc (a |-> sa)
  ensures   on gpu_loc (a |-> (lseq_map2 f sa sb <: lseq et lena))

(* 1-D gather: out[i] := src[idx[i]].  The idx values are modeled as u32 so
   they can be converted to Kuiper's 32-bit size_t for the device read. *)
let lseq_gather
  (#et : Type0)
  (#lens #leni : nat)
  (src : lseq et lens)
  (idx : (si0:lseq u32 leni { forall (j : natlt leni). FStar.UInt32.v (si0 @! j) < lens }))
  : GTot (lseq et leni)
  = Seq.init_ghost leni (fun i -> src @! FStar.UInt32.v (idx @! i))

inline_for_extraction noextract
fn gather_gpu
  (#et : Type0)
  (lens : szp)
  (leni : szp { leni <= max_blocks * max_threads })
  (#ls : Array1.layout lens) {| ctlayout ls |}
  (#li : Array1.layout leni) {| ctlayout li |}
  (#lo : Array1.layout leni) {| ctlayout lo |}
  (src : Array1.t et ls)
  (idx : Array1.t u32 li)
  (out : Array1.t et lo)
  (#_ : squash (Array1.is_global src))
  (#_ : squash (Array1.is_global idx))
  (#_ : squash (Array1.is_global out))
  (#ss : erased (lseq et lens))
  (#si : erased (lseq u32 leni))
  (#so : erased (lseq et leni))
  (#fs #fi : perm)
  norewrite
  preserves cpu ** on gpu_loc (src |-> Frac fs ss) ** on gpu_loc (idx |-> Frac fi si)
  requires  on gpu_loc (out |-> so)
  ensures   exists* so'. on gpu_loc (out |-> so')
