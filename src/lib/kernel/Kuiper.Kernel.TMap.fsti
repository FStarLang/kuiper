module Kuiper.Kernel.TMap

#lang-pulse

open Kuiper
open Kuiper.Tensor
module SZ = Kuiper.SizeT

inline_for_extraction noextract
noeq
type cshape : (#r : erased nat) -> (d : idesc r) -> Type =
  | CNil : cshape INil
  | CCons :
    (#r : erased nat) ->
    (#h : erased pos) -> ch : sz{SZ.v ch == reveal h} ->
    (#t : idesc r) -> cshape t ->
    cshape (ICons h t)

inline_for_extraction noextract
fn map_gpu
  (#et : Type0) (#r : erased nat) (#d : idesc r) (cd : cshape d)
  (f : et -> et)
  (#l : tlayout d) {| ctlayout l |}
  (n : sz{SZ.v n == sizeof d /\ n <= max_blocks * max_threads})
  (a : tensor et l { is_global a })
  (#s : chest d et)
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> chest_map f s)
