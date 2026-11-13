module Kuiper.Tensor.Results
#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Send

module RT = Kuiper.Tensor.Runtime
module KL = Kuiper.Locs
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Pledge = Pulse.Lib.Pledge

inline_for_extraction noextract
let _ = ()

(* A name for the actual native four-element std::vector<torch::Tensor>,
   not a pure replacement implementation. The type parameters describe its
   ordered Tensor cells; they are not four different C++ vector types. *)
val vector4 (a b c d:Type0) : Type0
val first (#a #b #c #d:Type0) (value:vector4 a b c d) : GTot (RT.tensor a)
val second (#a #b #c #d:Type0) (value:vector4 a b c d) : GTot (RT.tensor b)
val third (#a #b #c #d:Type0) (value:vector4 a b c d) : GTot (RT.tensor c)
val fourth (#a #b #c #d:Type0) (value:vector4 a b c d) : GTot (RT.tensor d)

val construction_error : Type0

noeq type construction_result (a b c d:Type0) =
  | Failed : error:construction_error -> construction_result a b c d
  | Constructed : value:vector4 a b c d -> construction_result a b c d

(* EXTERNAL NATIVE TRUST: outcome abstraction of the existing owning return
   through an initializer-list vector in the pinned C++20/libstdc++/PyTorch
   environment. Not a general alias-exporting vector-copy API, an extracted
   exception-catching shim, or a numerical postcondition.

   Constructed names the actual vector, in source order, after destruction of
   initializer-list temporaries and the four now-nonfinal output locals. The
   same exclusive storage-generation owner capabilities transfer to the result;
   they are neither copied nor created. Those nonfinal drops are not RT.release.
   Other local/input references are outside this operation's scope.

   Failed retains the original exception identity and the four original local
   owners. Partial container/temporary copies have been unwound. The caller
   must still prove final local-reference release and any scratch/input cleanup
   at THIS epoch. Neither outcome assumes host allocation success.

   Constructor copies touch native references, not Tensor storage or GPU work.
   Both outcomes preserve the exact pending pledge, stream tail and frame.
   No epoch_done, GPU cell permission, initialization or mathematical result
   is supplied. The independently owned RT.empty tensors cannot be fabricated
   from borrowed references or aliased pure Tensor names. *)
noextract
fn return_four
  (#a #b #c #d:Type0)
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (v0:RT.tensor a) (v1:RT.tensor b) (v2:RT.tensor c) (v3:RT.tensor d)
  (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch **
    RT.owner stream v0 ** RT.owner stream v1 **
    RT.owner stream v2 ** RT.owner stream v3 **
    Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
  returns outcome:construction_result a b c d
  ensures pure (
    match outcome with
    | Failed _ -> True
    | Constructed value ->
      first value == v0 /\ second value == v1 /\
        third value == v2 /\ fourth value == v3)
