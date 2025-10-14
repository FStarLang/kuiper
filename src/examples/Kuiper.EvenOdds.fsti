module Kuiper.EvenOdds

(* Some common definitions for the even/odds example. *)

#lang-pulse

open Kuiper
open Kuiper.Bijection

noextract
let bij_nat_interleave (#len:nat)
  : Tot (natlt len =~ either (natlt ((len + 1) / 2)) (natlt (len / 2))) =
  mk_bijection #(natlt len) #(either (natlt ((len + 1) / 2)) (natlt (len / 2)))
    (fun i -> if i % 2 = 0 then Inl (i / 2) else Inr (i / 2))
    (function
     | Inl i -> i * 2
     | Inr i -> 1 + i * 2)
    ez ez

noextract
let seq_evens #a (#n : nat)
  (s : lseq a n)
  : Tot (lseq a ((n+1)/2))
=
  Seq.init ((n+1)/2) fun i -> Seq.index s (2 * i)

noextract
let seq_odds #a (#n : nat)
  (s : lseq a n)
  : Tot (lseq a (n/2))
=
  Seq.init (n/2) fun i -> Seq.index s (2 * i + 1)

noextract
let seq_interleave #a (#n : nat)
  (s1 : lseq a ((n + 1) / 2))
  (s2 : lseq a (n / 2))
  : Tot (lseq a n)
=
  Seq.init n fun i ->
    if i % 2 = 0 then
      Seq.index s1 (i / 2)
    else
      Seq.index s2 (i / 2)

val interleave_lemma1 #a #n (s : lseq a n)
  : Lemma (ensures seq_interleave (seq_evens s) (seq_odds s) == s)
          [SMTPat (seq_interleave (seq_evens s) (seq_odds s))]

val interleave_lemma2 #a (#n:nat) (s1 : lseq a ((n + 1) / 2)) (s2 : lseq a (n / 2))
  : Lemma (ensures seq_evens (seq_interleave s1 s2) == s1 /\ seq_odds (seq_interleave s1 s2) == s2)
          [SMTPat (seq_evens (seq_interleave s1 s2)); SMTPat (seq_odds (seq_interleave s1 s2))]

noextract
let bij_seq_interleave (#et:Type) (#len:nat)
  : Tot (lseq et len =~ (lseq et ((len + 1) / 2)) & (lseq et (len / 2))) =
  mk_bijection #(lseq et len) #((lseq et ((len + 1) / 2)) & (lseq et (len / 2)))
    (fun i -> (seq_evens i, seq_odds i))
    (fun (i, j) -> seq_interleave i j)
    ez ez
