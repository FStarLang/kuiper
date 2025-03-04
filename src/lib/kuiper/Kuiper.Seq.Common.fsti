module Kuiper.Seq.Common

open FStar.Seq
open Kuiper.Functions
open Kuiper.Monoid

unfold
let ( @! ) (#a:Type) (s : seq a) (i : nat { i < Seq.length s }) : a = Seq.index #a s i

let rec seq_fold_left (#t:Type) (f: t -> t -> t) (acc: t) (v: seq t)
: GTot t (decreases length v)
=
  if length v = 0 then
    acc
  else
    let hd = head v in
    let tl = tail v in
    seq_fold_left f (f acc hd) tl

val lemma_seq_fold_left_sum (#a:Type) (e:a) (f: a -> a -> a)
  (s1 s2 : seq a)
  : Lemma (requires is_monoid e f)
          (ensures seq_fold_left f e (append s1 s2)
                   == f (seq_fold_left f e s1) (seq_fold_left f e s2))

val lem_append_slice (#a:Type) (s : seq a) (i j k : nat)
  : Lemma (requires i <= j /\ j <= k /\ k <= length s)
          (ensures append (slice s i j) (slice s j k) == slice s i k)

val lem_one_elem (#a:Type) (s : seq a) (v : a)
  : Lemma (requires length s == 1 /\ s @! 0 == v)
          (ensures s == seq![v])

let seq_replace
  (#a:Type)
  (s1 : seq a)
  (lo : nat)
  (hi : nat { lo <= hi /\ hi <= length s1 })
  (s2 : seq a { Seq.length s2 == hi - lo })
  : seq a
=
  let s1 = slice s1 0 lo in
  let s3 = slice s1 lo (Seq.length s1) in
  s1 ++ s2 ++ s3

let seq_blit
  (#a:Type)
  (s1 : seq a) (off1 : nat)
  (s2 : seq a) (off2 : nat)
  (cnt : nat{off1 + cnt <= length s1 /\ off2 + cnt <= length s2})
  : seq a
=
  let cut = slice s2 off2 (off2 + cnt) in
  seq_replace s1 off1 (off1 + cnt) cut
