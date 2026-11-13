module KPlay.Proof.Base.SourceArrays

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
module A = Pulse.Lib.Array
module C = Kuiper.Chest
module T = Kuiper.Tensor
module BF = Kuiper.BFloat16
module Native = KPlay.Proof.Base.NativeProjection

inline_for_extraction noextract
let flat_weights (value:C.chest2 BF.t 752393024 1)
  : Lemma (T.to_seq Native.weight_layout value == Native.packed_values value) =
  C.ext (T.from_seq Native.weight_layout (Native.packed_values value)) value;
  T.to_from Native.weight_layout (Native.packed_values value)

ghost
fn weight_to_array (weights:array2 BF.t Native.weight_layout)
  (#value:C.chest2 BF.t 752393024 1) (#fraction:perm)
  requires weights |-> Frac fraction value
  ensures A.pts_to (T.core weights) #fraction (Native.packed_values value)
{
  flat_weights value;
  T.tensor_concr weights;
  rewrite A.pts_to (T.core weights) #fraction (T.to_seq Native.weight_layout value)
    as A.pts_to (T.core weights) #fraction (Native.packed_values value);
}

ghost
fn array_to_weight (weights:array2 BF.t Native.weight_layout)
  (#value:C.chest2 BF.t 752393024 1) (#fraction:perm)
  requires A.pts_to (T.core weights) #fraction (Native.packed_values value)
  ensures weights |-> Frac fraction value
{
  flat_weights value;
  rewrite A.pts_to (T.core weights) #fraction (Native.packed_values value)
    as A.pts_to (T.core weights) #fraction (T.to_seq Native.weight_layout value);
  T.tensor_abs Native.weight_layout (T.core weights) #fraction #value;
  T.lem_core_from_array weights;
  rewrite each (T.from_array Native.weight_layout (T.core weights)) as weights;
}
