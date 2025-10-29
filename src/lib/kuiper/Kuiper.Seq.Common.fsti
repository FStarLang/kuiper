module Kuiper.Seq.Common

open FStar.Seq
open Kuiper.Functions
open Kuiper.Monoid

type seq_view a =
  | SNil
  | SCons : a -> seq a -> seq_view a

let pack_seq (v : seq_view 'a) : seq 'a =
  match v with
  | SNil -> Seq.empty
  | SCons x xs -> Seq.cons x xs

(* Mark as coercion? *)
let view_seq (s : seq 'a) : v:(seq_view 'a){pack_seq v == s} =
  if Seq.length s = 0
  then  (
    assert (Seq.equal s Seq.empty);
    SNil
  ) else SCons (Seq.head s) (Seq.tail s)

unfold
let ( @! ) (#a:Type) (s : seq a) (i : nat { i < Seq.length s }) : a = Seq.index #a s i

unfold
let ( @+ ) (#a:Type) (s1 s2 : seq a) : seq a = Seq.append s1 s2

let rec seq_fold_left (#a #b : Type) (f: b -> a -> b) (acc: b) (v: seq a)
  : GTot b (decreases length v)
  = match view_seq v with
    | SNil -> acc
    | SCons hd tl -> seq_fold_left f (f acc hd) tl

let rec seq_fold_right (#a #b : Type) (f: a -> b -> b) (v : seq a) (e : b)
  : GTot b (decreases length v)
  = match view_seq v with
    | SNil -> e
    | SCons hd tl -> f hd (seq_fold_right f tl e)

val lemma_seq_fold_right_sum (#a:Type) (e:a) (f: a -> a -> a)
  (s1 s2 : seq a)
  : Lemma (requires is_monoid e f)
          (ensures seq_fold_right f (append s1 s2) e
                   == seq_fold_right f s1 e `f` seq_fold_right f s2 e)

val lemma_seq_fold_left_sum (#a:Type) (e:a) (f: a -> a -> a)
  (s1 s2 : seq a)
  : Lemma (requires is_monoid e f)
          (ensures seq_fold_left f e (append s1 s2)
                   == seq_fold_left f e s1 `f` seq_fold_left f e s2)

val lem_append_slice (#a:Type) (s : seq a) (i j k : nat)
  : Lemma (requires i <= j /\ j <= k /\ k <= length s)
          (ensures append (slice s i j) (slice s j k) == slice s i k)

val lem_one_elem (#a:Type) (s : seq a) (v : a)
  : Lemma (requires length s == 1 /\ s @! 0 == v)
          (ensures s == seq![v])
          [SMTPat (length s); SMTPat (seq![v])] // not sure this actually triggers

val lemma_upd_index (#a:Type) (s : seq a) (i : nat{i < Seq.length s})
  : Lemma (requires True)
          (ensures (Seq.upd s i (Seq.index s i)) == s)
          [SMTPat (Seq.upd s i (Seq.index s i))]

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

let seq_take
  (#a:Type)
  (n : nat)
  (s : seq a{n <= length s})
  : seq a
=
  slice s 0 n

let seq_drop
  (#a:Type)
  (n : nat)
  (s : seq a{n <= length s})
  : seq a
=
  slice s n (length s)
