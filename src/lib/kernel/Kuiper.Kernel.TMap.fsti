module Kuiper.Kernel.TMap

(* Pointwise map of a (pure) function on a tensor.

Internally, there is infra to map stateful functions (that can read from other
tensors, etc). We could expose that. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Shareable
module SZ = Kuiper.SizeT

inline_for_extraction noextract
val kmap
  (#et : Type0) (#r : erased nat) (#d : shape r) (cd : cshape d)
  (frame : perm -> slprop) {| shareable frame |}
  (vf : abs d -> et -> et -> prop) // spec for f
  (f :
    fn (#fr: perm) (i : conc d) (x : et)
      preserves frame fr
      returns r : et
      ensures pure (vf (up i) x r))
  (#l : tlayout d) {| ctlayout l |}
  (n : sz{SZ.v n == sizeof d /\ n <= max_blocks * max_threads /\ n > 0}) // n > 0 cause of fraction stuff
  (a : tensor et l)
  (#s : chest d et)
  (#_ : is_global a)
  (#fr: perm)
  : kernel_desc
      (requires frame fr ** a |-> s)
      (ensures  frame fr ** exists* s'. a |-> s' **
        pure (chest_foralli (fun i x -> vf i (acc s i) x) s'))

inline_for_extraction noextract
fn map_gpu
  (#et : Type0) (#r : erased nat) (#d : shape r) (cd : cshape d)
  (f : et -> et)
  (#l : tlayout d) {| ctlayout l |}
  (n : sz{SZ.v n == sizeof d /\ n <= max_blocks * max_threads /\ n > 0})
  (a : tensor et l { is_global a })
  (#s : chest d et)
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> chest_map f s)
