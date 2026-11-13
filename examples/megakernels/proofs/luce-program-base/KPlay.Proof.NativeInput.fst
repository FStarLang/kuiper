module KPlay.Proof.NativeInput
#lang-pulse

open Pulse.Lib.Core
module Input = Kuiper.Tensor.Input
module RT = Kuiper.Tensor.Runtime
module A = Pulse.Lib.Array
module KA = Kuiper.Array.Core
module KL = Kuiper.Locs
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module Storage = KPlay.Proof.Shared.Storage

noextract
noeq type arguments (p:pos{p <= 520}) = {
  weights: value:Input.argument BF.t{Input.dimensions value == [752393024]};
  prompt: value:Input.argument I64.t{Input.dimensions value == [p]};
}

inline_for_extraction noextract
let cell (#a:Type0) (value:Input.argument a)
  (index:nat{index < RT.numel (Input.dimensions value)}) : GTot (Storage.cell a) =
  Input.view_metadata value;
  {buffer = Input.data value; offset = index}

inline_for_extraction noextract
let cell_binding (#a:Type0) (value:Input.argument a)
  (index:nat{index < RT.numel (Input.dimensions value)})
  : Lemma (
      (cell value index).buffer == Input.data value /\
      (cell value index).offset == index /\
      A.base_of (cell value index).buffer == A.base_of (Input.backing value) /\
      A.offset_of (cell value index).buffer + (cell value index).offset ==
        A.offset_of (Input.backing value) + Input.offset value + index) =
  Input.view_metadata value

inline_for_extraction noextract
let pointer_result (#p:pos{p <= 520}) (inputs:arguments p)
  (addresses:(A.array BF.t & A.array I64.t & I64.t)) : prop =
  addresses._1 == Input.data inputs.weights /\
  addresses._2 == Input.data inputs.prompt /\
  I64.v addresses._3 == p /\
  KA.is_global_array addresses._1 /\ KA.is_global_array addresses._2 /\
  A.length addresses._1 == 752393024 /\ A.length addresses._2 == p

inline_for_extraction noextract
let source_post (#p:pos{p <= 520}) (inputs:arguments p) (frame:slprop)
  (addresses:(A.array BF.t & A.array I64.t & I64.t)) : slprop =
  frame ** pure (pointer_result inputs addresses)

(* The references come from the actual native parameter scopes, not from frame.
   frame may contain the unchanged SharedSpec input-cell predicate, epochs and
   arbitrary outside resources. No GPU permission is created or reinterpreted. *)
noextract
fn source_pointer_queries (#p:pos{p <= 520}) (inputs:arguments p) (frame:slprop)
  preserves KL.cpu ** frame
  returns addresses:(A.array BF.t & A.array I64.t & I64.t)
  ensures pure (pointer_result inputs addresses)
{
  let addresses = Input.with_parameter #BF.t #(A.array BF.t & A.array I64.t & I64.t)
    inputs.weights frame (source_post inputs frame) fn () {
    Input.with_parameter #I64.t #(A.array BF.t & A.array I64.t & I64.t) inputs.prompt
      (Input.reference inputs.weights ** frame)
      (fun addresses -> Input.reference inputs.weights ** source_post inputs frame addresses)
      fn () {
        let weights = Input.data_ptr inputs.weights;
        let prompt = Input.data_ptr inputs.prompt;
        let count = Input.numel inputs.prompt;
        Input.view_metadata inputs.weights;
        Input.view_metadata inputs.prompt;
        assert pure (pointer_result inputs (weights, prompt, count));
        fold source_post inputs frame (weights, prompt, count);
        (weights, prompt, count)
      }
  };
  unfold source_post inputs frame addresses;
  addresses
}
