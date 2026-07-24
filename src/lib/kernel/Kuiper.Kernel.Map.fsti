module Kuiper.Kernel.Map

#lang-pulse

open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
module SZ = Kuiper.SizeT

unfold let chest1_map2
  (#et : Type0) (#len : nat)
  (f : et -> et -> et)
  (a b : chest1 et len)
  : chest1 et len
  = mk1 (fun i -> f (acc1 a i) (acc1 b i))

unfold let chest1_map3
  (#et : Type0) (#len : nat)
  (f : et -> et -> et -> et)
  (a b c : chest1 et len)
  : chest1 et len
  = mk1 (fun i -> f (acc1 a i) (acc1 b i) (acc1 c i))

inline_for_extraction noextract
fn map_gpu
  (#et : Type0)
  (f : et -> et)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#s : chest1 et lena)
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> chest_map f s)

inline_for_extraction noextract
fn map_gpu_to
  (#it #ot : Type0)
  (f : it -> ot)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#li : layout1 lena) {| ctlayout li |}
  (#lo : layout1 lena) {| ctlayout lo |}
  (input : array1 it li { is_global input })
  (output : array1 ot lo { is_global output })
  (#si : chest1 it lena)
  (#so : chest1 ot lena)
  (#fi : perm)
  norewrite
  preserves cpu ** on gpu_loc (input |-> Frac fi si)
  requires on gpu_loc (output |-> so)
  ensures  on gpu_loc (output |-> mk1 (fun i -> f (acc1 si i)))

inline_for_extraction noextract
fn map_gpu2
  (#et : Type0)
  (f : et -> et -> et)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (a : array1 et la { is_global a })
  (b : array1 et lb { is_global b })
  (#sa #sb : chest1 et lena)
  (#fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (b |-> Frac fb sb)
  requires on gpu_loc (a |-> sa)
  ensures  on gpu_loc (a |-> mk1 (fun i -> f (acc1 sa i) (acc1 sb i)))

inline_for_extraction noextract
fn map_gpu2_to
  (#at #bt #ot : Type0)
  (f : at -> bt -> ot)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (#lo : layout1 lena) {| ctlayout lo |}
  (a : array1 at la { is_global a })
  (b : array1 bt lb { is_global b })
  (output : array1 ot lo { is_global output })
  (#sa : chest1 at lena)
  (#sb : chest1 bt lena)
  (#so : chest1 ot lena)
  (#fa #fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb)
  requires on gpu_loc (output |-> so)
  ensures  on gpu_loc (output |-> mk1 (fun i -> f (acc1 sa i) (acc1 sb i)))

inline_for_extraction noextract
fn map_gpu3
  (#et : Type0)
  (f : et -> et -> et -> et)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (#lc : layout1 lena) {| ctlayout lc |}
  (a : array1 et la { is_global a })
  (b : array1 et lb { is_global b })
  (c : array1 et lc { is_global c })
  (#sa #sb #sc : chest1 et lena)
  (#fb #fc : perm)
  norewrite
  preserves cpu ** on gpu_loc (b |-> Frac fb sb) ** on gpu_loc (c |-> Frac fc sc)
  requires on gpu_loc (a |-> sa)
  ensures  on gpu_loc (a |-> mk1 (fun i -> f (acc1 sa i) (acc1 sb i) (acc1 sc i)))

inline_for_extraction noextract
fn map_gpu3_to
  (#at #bt #ct #ot : Type0)
  (f : at -> bt -> ct -> ot)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (#lc : layout1 lena) {| ctlayout lc |}
  (#lo : layout1 lena) {| ctlayout lo |}
  (a : array1 at la { is_global a })
  (b : array1 bt lb { is_global b })
  (c : array1 ct lc { is_global c })
  (output : array1 ot lo { is_global output })
  (#sa : chest1 at lena)
  (#sb : chest1 bt lena)
  (#sc : chest1 ct lena)
  (#so : chest1 ot lena)
  (#fa #fb #fc : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb) ** on gpu_loc (c |-> Frac fc sc)
  requires on gpu_loc (output |-> so)
  ensures  on gpu_loc (output |-> mk1 (fun i -> f (acc1 sa i) (acc1 sb i) (acc1 sc i)))

inline_for_extraction noextract
fn map_host
  (#et : Type0) {| sized et |}
  (f : et -> et)
  (lena : szp { lena <= max_blocks * max_threads /\ lena > 0 })
  (a : Pulse.Lib.Vec.lvec et lena)
  (#s : erased (lseq et lena))
  preserves cpu
  requires a |-> s
  ensures  a |-> lseq_map f s

let mapi_value
  (#et : Type0) (#len : nat { SZ.fits len })
  (f : et -> (i : SZ.t { SZ.v i < len }) -> et)
  (x : et) (i : natlt len)
  : et
  = f x (SZ.uint_to_t i)

let map_chest1i
  (#et : Type0) (#len : nat { SZ.fits len })
  (f : et -> (i : SZ.t { SZ.v i < len }) -> et)
  (s : chest1 et len)
  : chest1 et len
  = Kuiper.Chest.chest1_mapi (fun i x -> mapi_value f x i) s

inline_for_extraction noextract
fn mapi_gpu
  (#et : Type0)
  (lena : szp { lena <= max_blocks * max_threads /\ lena <= 2147483648 /\ lena > 0 /\ SZ.fits lena })
  (f : et -> (i : SZ.t { SZ.v i < lena }) -> et)
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#s : chest1 et lena)
  preserves cpu
  requires on gpu_loc (a |-> s)
  ensures  on gpu_loc (a |-> map_chest1i f s)
