module Kuiper.Seq.Common

open FStar.Seq
open Kuiper.Functions
open Kuiper.Monoid
open Kuiper.Common

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
  then (
    assert (Seq.equal s Seq.empty);
    SNil
  ) else SCons (Seq.head s) (Seq.tail s)

unfold
let ( @! ) (#a:Type) (s : seq a) (i : nat { i < Seq.length s }) : a = Seq.index #a s i

unfold
let ( @+ ) (#a:Type) (s1 s2 : seq a) : seq a = Seq.append s1 s2

let seq_forall (#a : Type) (f: a -> prop) (s: seq a) : prop =
  forall (i : nat { i < Seq.length s }). f (s @! i)

let seq_forallb (#a : Type) (f: a -> GTot bool) (s: seq a) : prop =
  seq_forall (fun x -> f x) s

let seq_map (#a #b : Type) (f: a -> b) (s: seq a) : GTot (seq b) =
  Seq.init_ghost (Seq.length s) (fun i -> f (s @! i))

val seq_map_append (#a #b : Type) (f: a -> b) (s1 s2 : seq a)
  : Lemma (ensures seq_map f (s1 @+ s2) == seq_map f s1 @+ seq_map f s2)
          [SMTPat (seq_map f (s1 @+ s2))]

let lseq_map (#a #b : Type) (#len : nat) (f: a -> b) (s: lseq a len) : GTot (lseq b len) =
  seq_map f s

let lseq_upd (#a:Type) (#n:nat) (s : lseq a n) (i : nat { i < n }) (v : a)
  : GTot (lseq a n)
  = Seq.upd s i v

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

val lemma_seq_fold_left_slice (#a #b:Type) (e:b) (f: b -> a -> b)
  (s : seq a) (i j : nat)
  : Lemma (requires i <= j /\ j < length s)
          (ensures seq_fold_left f e (slice s i (j + 1))
                    == seq_fold_left f e (slice s i j) `f` (s @! j))
          [SMTPat (seq_fold_left f e (slice s i (j + 1)))]

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
  : Pure (seq a)
         (requires True)
         (ensures fun s1' -> Seq.length s1' == Seq.length s1)
=
  let left  = slice s1 0 lo in
  let right = slice s1 hi (Seq.length s1) in
  left ++ s2 ++ right

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


// Unfortunate to have to define and use this
let seq_refine #a (p : a -> prop)
  (s : seq a { forall i. p (s @! i) })
  : GTot (lseq (x:a{p x}) (Seq.length s))
  = Seq.init_ghost #(x:a{p x}) (Seq.length s) (fun i -> s @! i)

val lem_seq_refine_at #a (p : a -> prop)
  (s : seq a { forall i. p (s @! i) })
  (i : nat {i < Seq.length s})
  : Lemma ((seq_refine p s) @! i == s @! i)
          // [SMTPat (seq_refine #a p s @! i)]
          // ^ Does not seem to work (warns)

let seq_stride_length (#a:Type)
  (s : seq a) (stride : pos) (off : natlt stride)
  : GTot nat
  = (Seq.length s - off + stride - 1) / stride

let seq_stride (#a:Type)
  (s : seq a) (stride : pos) (off : natlt stride)
  : GTot (seq a)
  = Seq.init_ghost
      (seq_stride_length s stride off)
      (fun i -> s @! (off + i * stride))
