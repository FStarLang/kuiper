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

(* Three-array elementwise map into a separate output array, each array with a
   possibly different element type: o[i] := f a[i] b[i] c[i].
   The inputs a, b, c are held read-only; o receives the result. *)

let lseq_map3
  (#a #b #c #d : Type0)
  (#len : nat)
  (f : a -> b -> c -> d)
  (sa : lseq a len) (sb : lseq b len) (sc : lseq c len)
  : GTot (lseq d len)
  = Seq.init_ghost len (fun i -> f (Seq.index sa i) (Seq.index sb i) (Seq.index sc i))

inline_for_extraction noextract
fn map_gpu3
  (#eta #etb #etc #eto : Type0)
  (f : eta -> etb -> etc -> eto)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lb : Array1.layout lena) {| ctlayout lb |}
  (#lc : Array1.layout lena) {| ctlayout lc |}
  (#lo : Array1.layout lena) {| ctlayout lo |}
  (a : Array1.t eta la)
  (b : Array1.t etb lb)
  (c : Array1.t etc lc)
  (o : Array1.t eto lo)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array1.is_global b))
  (#_ : squash (Array1.is_global c))
  (#_ : squash (Array1.is_global o))
  (#sa : erased (lseq eta lena))
  (#sb : erased (lseq etb lena))
  (#sc : erased (lseq etc lena))
  (#so : erased (lseq eto lena))
  (#fa #fb #fc : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb) ** on gpu_loc (c |-> Frac fc sc)
  requires  on gpu_loc (o |-> so)
  ensures   on gpu_loc (o |-> (lseq_map3 f sa sb sc <: lseq eto lena))
