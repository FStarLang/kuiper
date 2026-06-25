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

(* In-place map with index: a[i] := f a[i] i.  Index is passed as a runtime
   SZ.t value (a stateful kernel body cannot consume a ghost nat). *)

let lseq_mapi
  (#et : Type0)
  (#len : nat { SZ.fits len })
  (f : et -> (i:SZ.t { SZ.v i < len }) -> et)
  (s : lseq et len)
  : GTot (lseq et len)
  = Seq.init_ghost len (fun (i : natlt len) -> f (Seq.index s i) (SZ.uint_to_t i))

inline_for_extraction noextract
fn mapi_gpu
  (#et : Type0)
  (lena : szp { lena <= max_blocks * max_threads })
  (f : et -> (i:SZ.t { SZ.v i < lena }) -> et)
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#s : erased (lseq et lena))
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> lseq_mapi f s)

(* Pointwise map from one array to another with possibly different element types.
   c[i] := f a[i]. *)

inline_for_extraction noextract
fn map_gpu_notinplace
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
  ensures   on gpu_loc (c |-> (lseq_map f sa <: lseq ot lena))

(* Two-array elementwise map: a[i] := f a[i] b[i]. *)

let lseq_map2
  (#a #b #c : Type0)
  (#len : nat)
  (f : a -> b -> c)
  (sa : lseq a len) (sb : lseq b len)
  : GTot (lseq c len)
  = Seq.init_ghost len (fun i -> f (Seq.index sa i) (Seq.index sb i))

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
