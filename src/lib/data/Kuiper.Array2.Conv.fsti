module Kuiper.Array2.Conv

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Shape
open Kuiper.EMatrix

module A2 = Kuiper.Array2

(* Bridge a rank-2 tensor's gpu points-to into the equivalent Array2 view. *)
inline_for_extraction noextract
fn array2_of_tensor
  (#et:Type0) (#rows #cols:nat) (#l : A2.layout rows cols)
  (g : tensor et l { is_global g })
  (#f:perm) (#s : ematrix et rows cols)
  requires on gpu_loc (g |-> Frac f s)
  returns a : (a : A2.t et l { A2.is_global a })
  ensures on gpu_loc (a |-> Frac f s) ** pure (A2.as_tensor a == g)

(* Bridge an Array2's gpu points-to back into the equivalent tensor view. *)
inline_for_extraction noextract
fn tensor_of_array2
  (#et:Type0) (#rows #cols:nat) (#l : A2.layout rows cols)
  (a : A2.t et l { A2.is_global a })
  (#f:perm) (#s : ematrix et rows cols)
  requires on gpu_loc (a |-> Frac f s)
  returns g : (g : tensor et l { is_global g })
  ensures on gpu_loc (g |-> Frac f s) ** pure (g == A2.as_tensor a)
