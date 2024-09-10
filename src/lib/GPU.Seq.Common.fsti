module GPU.Seq.Common

open FStar.Seq

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

let is_associative (#a:Type) (f : a -> a -> a) : prop =
  forall x y z. f (f x y) z == f x (f y z)

let is_neutral_for (#a:Type) (e : a) (f : a -> a -> a) : prop =
  forall x. f e x == x /\ f x e == x

let is_monoid (#a:Type) (e : a) (f : a -> a -> a) : prop =
  is_associative f /\ is_neutral_for e f

val lemma_seq_fold_left_sum (#a:Type) (e:a) (f: a -> a -> a)
  (s1 s2 : seq a)
  : Lemma (requires is_monoid e f)
          (ensures seq_fold_left f e (append s1 s2)
                   == f (seq_fold_left f e s1) (seq_fold_left f e s2))
