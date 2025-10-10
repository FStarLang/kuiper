module Kuiper.EvenOdds

(* Some common definitions for the even/odds example. *)

#lang-pulse

open Kuiper
open Kuiper.Bijection

let interleave_lemma1 #a #n (s : lseq a n)
  : Lemma (ensures seq_interleave (seq_evens s) (seq_odds s) == s)
= assert (Seq.equal (seq_interleave (seq_evens s) (seq_odds s)) s);
  ()

let interleave_lemma2 #a (#n:nat) (s1 : lseq a ((n + 1) / 2)) (s2 : lseq a (n / 2))
  : Lemma (ensures seq_evens (seq_interleave s1 s2) == s1 /\ seq_odds (seq_interleave s1 s2) == s2)
= assert (Seq.equal (seq_evens (seq_interleave s1 s2)) s1);
  assert (Seq.equal (seq_odds (seq_interleave s1 s2)) s2);
  ()
