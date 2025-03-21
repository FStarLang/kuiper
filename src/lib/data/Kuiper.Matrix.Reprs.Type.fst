module Kuiper.Matrix.Reprs.Type
#lang-pulse

open Kuiper
open Kuiper.Bijection
open FStar.Tactics.Typeclasses
module SZ = FStar.SizeT

#push-options "--warn_error -288"
let clayout_fits (#rows #cols : nat) (#l : mlayout rows cols)
  (c : clayout l)
  : Lemma (SZ.fits (mlayout_size l))
  = admit()
    (* I feel this should be provable: c_to is injective and has rows * cols distinct
    arguments, and returns size_t's (that fit). *)
#pop-options
