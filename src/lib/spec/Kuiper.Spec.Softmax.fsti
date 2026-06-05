module Kuiper.Spec.Softmax

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Seq.Common

(* Real-value golden specification for softmax. *)
let softmax_real (s : seq real) =
  seq_mapi s (fun x i ->
    let exps = seq_map rexp s in
    let summ : real = rsum exps in
    rexp x /. summ)

val shift_denom (r0 : Seq.seq real) (c : real)
  : Lemma (ensures rsum (seq_map (fun z -> rexp (z -. c)) r0)
                   == rsum (seq_map rexp r0) /. rexp c)

(* Softmax is unchanged by a shift. *)
val softmax_shift (r0 : seq real) (c : real)
  : Lemma (ensures softmax_real (seq_map (fun x -> x -. c) r0)
                   == softmax_real r0)
