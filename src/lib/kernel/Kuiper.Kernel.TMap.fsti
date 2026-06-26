module Kuiper.Kernel.TMap

(* Pointwise map of a (pure) function on a tensor.

Internally, there is infra to map stateful functions (that can read from other
tensors, etc). We could expose that. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn map_gpu
  (#et : Type0) (#r : erased nat) (#d : shape r) (cd : cshape d)
  (f : et -> et)
  (#l : tlayout d) {| ctlayout l |}
  (n : sz{SZ.v n == sizeof d /\ n <= max_blocks * max_threads})
  (a : tensor et l { is_global a })
  (#s : chest d et)
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> chest_map f s)
