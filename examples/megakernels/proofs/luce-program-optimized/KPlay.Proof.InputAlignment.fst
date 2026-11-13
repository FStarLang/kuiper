module KPlay.Proof.InputAlignment

module BF16 = Kuiper.BFloat16
module Sized = Kuiper.Sized
module SZ = Kuiper.SizeT
module Mem = Kuiper.Array.Core
module Native = Pulse.Lib.Array
module Divides = Kuiper.Divides
module Input = Kuiper.Tensor.Input
module Entry = KPlay.Proof.NativeInput
module Model = KPlay.Proof.Shared.Model

inline_for_extraction noextract
let weight_count : pos = 752393024

(* reduced.cu:183-193, layer=0 and row=0; dot_bf16 starts at lane=k=0. *)
inline_for_extraction noextract
let first_projection_index : nat = 254280704 + 11012096

inline_for_extraction noextract
let smallest_call : Model.invocation = {
  Model.smoke = true;
  Model.prompt_length = 1;
  Model.decode_steps = 1;
}

inline_for_extraction noextract
let first_projection_domain ()
  : Lemma (
      Model.calls smallest_call == 1 /\
      first_projection_index == 265292800 /\
      first_projection_index % 8 == 0 /\
      first_projection_index + 1024 <= weight_count /\
      1 + weight_count == 752393025 /\
      0 % 4 <> 3 /\ 0 < 6144 /\ 0 < 32 /\ 0 < 1024) = ()

inline_for_extraction noextract
let bf16_bytes () : GTot nat = SZ.v (Sized.size #BF16.t)

inline_for_extraction noextract
let view_load_address (backing_address storage_offset element:nat) : GTot nat =
  backing_address + (storage_offset + element) * bf16_bytes ()

(* This is integer address arithmetic, not a constructor for a native Tensor.
   TensorImpl.h:data_ptr_impl_impl supplies the storage-offset correspondence. *)
inline_for_extraction noextract
let offset_one_first_load (backing_address:nat{backing_address % 16 == 0})
  : Lemma (
      view_load_address backing_address 1 first_projection_index ==
        backing_address + 530585602 /\
      view_load_address backing_address 1 first_projection_index % 16 == 2 /\
      view_load_address backing_address 1 first_projection_index % 2 == 0 /\
      view_load_address backing_address 0 first_projection_index % 16 == 0 /\
      view_load_address backing_address 8 first_projection_index % 16 == 0) =
  BF16.lem_sizeof ()

inline_for_extraction noextract
let dot_chunks_preserve_offset_phase
  (backing_address:nat{backing_address % 16 == 0})
  (offset:nat{offset % 8 == 0})
  (lane:nat{lane < 32}) (chunk:nat)
  : Lemma (
      view_load_address backing_address 1 (offset + lane * 8 + chunk * 256) % 16 == 2) =
  BF16.lem_sizeof ()

inline_for_extraction noextract
let first_projection_input_cell (inputs:Entry.arguments 1)
  : Lemma (
      let at = Entry.cell inputs.weights first_projection_index in
      at.buffer == Input.data inputs.weights /\
      at.offset == first_projection_index /\
      Native.base_of at.buffer == Native.base_of (Input.backing inputs.weights) /\
      Native.offset_of at.buffer + at.offset ==
        Native.offset_of (Input.backing inputs.weights) +
          Input.offset inputs.weights + first_projection_index) =
  Entry.cell_binding inputs.weights first_projection_index

(* Instantiation of the actual array API's alignment predicate. No implication
   from Input.reference, initialized cells or contiguity to alignment is used. *)
inline_for_extraction noextract
let first_projection_not_vector_aligned (weights:Native.array BF16.t)
  : Lemma
      (requires Mem.base_address weights % 16 == 2)
      (ensures not (Mem.aligned' 16 weights first_projection_index)) =
  BF16.lem_sizeof ();
  Divides.lemma_divides_mod 16 (Mem.base_address weights + first_projection_index * 2);
  Divides.lemma_nat_divides_pos_divides 16
    (Mem.base_address weights + first_projection_index * 2)

inline_for_extraction noextract
let source_first_projection_alignment_obstruction (inputs:Entry.arguments 1)
  : Lemma
      (requires Mem.base_address (Input.data inputs.weights) % 16 == 2)
      (ensures
        (Entry.cell inputs.weights first_projection_index).buffer ==
          Input.data inputs.weights /\
        (Entry.cell inputs.weights first_projection_index).offset ==
          first_projection_index /\
        first_projection_index + 8 <= weight_count /\
        not (Mem.aligned' 16 (Input.data inputs.weights) first_projection_index)) =
  first_projection_domain ();
  first_projection_input_cell inputs;
  first_projection_not_vector_aligned (Input.data inputs.weights)
