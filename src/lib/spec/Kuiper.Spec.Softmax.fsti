module Kuiper.Spec.Softmax

#lang-pulse
open Kuiper
open Kuiper.Chest
open Kuiper.Seq.Common

(* Real-value spec for softmax. *)
let softmax_real #n (s : chest1 real n) : chest1 real n =
  chest1_mapi (fun i x ->
    let exps = chest_map exp s in
    let summ : real = chest1_rsum exps in
    exp x /. summ) s

(* Seq-level mirror of softmax_real, for the lseq-based numeric proofs. *)
let softmax_real_seq (s : seq real) : GTot (seq real) =
  seq_mapi s (fun x i ->
    let exps = seq_map exp s in
    let summ : real = rsum exps in
    exp x /. summ)

(* Bridge: the chest1 spec is the seq spec transported through chest1_to_seq. *)
val lem_softmax_real_to_seq #n (s : chest1 real n)
  : Lemma (chest1_to_seq (softmax_real s) == softmax_real_seq (chest1_to_seq s))
          [SMTPat (chest1_to_seq (softmax_real s))]

val shift_denom (r0 : Seq.seq real) (c : real)
  : Lemma (ensures rsum (seq_map (fun z -> exp (z -. c)) r0)
                   == rsum (seq_map exp r0) /. exp c)

(* Softmax is unchanged by a shift. *)
val softmax_shift #n (r0 : chest1 real n) (c : real)
  : Lemma (ensures softmax_real (chest_map (fun x -> x -. c) r0)
                   == softmax_real r0)
