module Kuiper.Example.ArraySubviewResidency

#lang-pulse

open Kuiper

module A = Pulse.Lib.Array
module K = Kuiper.Array.Core

let window_metadata (#a:Type0) (arr:A.array a) (lo:nat)
  (hi:nat{lo <= hi /\ hi <= A.length arr})
  : Lemma (
      A.loc_id_of_array (A.gsub arr lo hi) == A.loc_id_of_array arr /\
      A.visibility_of_array (A.gsub arr lo hi) == A.visibility_of_array arr /\
      K.visibility_of (A.gsub arr lo hi) == K.visibility_of arr) = ()

let global_window (#a:Type0) (arr:A.array a{K.is_global_array arr}) (lo:nat)
  (hi:nat{lo <= hi /\ hi <= A.length arr})
  : Lemma (K.is_global_array (A.gsub arr lo hi)) = ()

let empty_and_full_windows (#a:Type0) (arr:A.array a{K.is_global_array arr})
  (at:nat{at <= A.length arr})
  : Lemma (
      K.is_global_array (A.gsub arr at at) /\
      K.is_global_array (A.gsub arr 0 (A.length arr))) = ()

let nested_window (#a:Type0) (arr:A.array a{K.is_global_array arr}) (lo:nat)
  (hi:nat{lo <= hi /\ hi <= A.length arr})
  (inner_lo:nat) (inner_hi:nat{inner_lo <= inner_hi /\ inner_hi <= hi - lo})
  : Lemma (
      K.is_global_array (A.gsub (A.gsub arr lo hi) inner_lo inner_hi) /\
      A.loc_id_of_array (A.gsub (A.gsub arr lo hi) inner_lo inner_hi) ==
        A.loc_id_of_array arr) = ()

[@@expect_failure [19]]
let cannot_invent_global_residency (#a:Type0) (arr:A.array a) (lo:nat)
  (hi:nat{lo <= hi /\ hi <= A.length arr})
  : Lemma (K.is_global_array (A.gsub arr lo hi)) = ()

[@@expect_failure [228]]
fn cannot_read_from_metadata (arr:larray u32 1{is_global_array arr})
  preserves gpu
  returns u32
{
  arr.(0sz)
}
