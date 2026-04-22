module Kuiper.Seq.Common

open FStar.Seq

let rec lemma_seq_fold_right_sum
  (#a:Type) (e:a) (f: a -> a -> a)
  (s1 s2 : seq a)
: Lemma
  (requires is_monoid e f)
  (ensures seq_fold_right f (append s1 s2) e
           == seq_fold_right f s1 e `f` seq_fold_right f s2 e)
  (decreases Seq.length s1)
= match view_seq s1 with
  | SNil ->
    assert (seq_fold_right f s1 e == e);
    assert (Seq.equal (append s1 s2) s2);
    ()
  | SCons hd tl ->
    let SCons hd' tl' = view_seq (append s1 s2) in
    assert (hd == hd');
    assert (tl' `Seq.equal` (append tl s2));
    calc (==) {
      seq_fold_right f (append s1 s2) e;
    (==) {}
      f hd (seq_fold_right f (append tl s2) e);
    (==) { lemma_seq_fold_right_sum e f tl s2 }
      f hd (seq_fold_right f tl e `f` seq_fold_right f s2 e);
    (==) {}
      (f hd (seq_fold_right f tl e)) `f` (seq_fold_right f s2 e);
    (==) {}
      seq_fold_right f s1 e `f` (seq_fold_right f s2 e);
    }

let rec seq_fold_left_append (#a #b:Type) (f: b -> a -> b) (acc:b) (l0 l1:seq a)
: Lemma
  (ensures seq_fold_left f acc (l0 `append` l1)
       == seq_fold_left f (seq_fold_left f acc l0) l1)
  (decreases Seq.length l0)
= match view_seq l0 with
  | SNil -> assert (Seq.equal (append l0 l1) l1)
  | SCons hd tl ->
    let SCons hd' tl' = view_seq (append l0 l1) in
    assert (hd == hd');
    assert (tl' `Seq.equal` (append tl l1));
    seq_fold_left_append f (f acc hd) tl l1

let rec lemma_seq_fold_left_sum'
  (#a:Type) (acc:a) (e:a) (f: a -> a -> a)
  (s1 s2 : seq a)
: Lemma
  (requires is_monoid e f)
  (ensures seq_fold_left f acc (append s1 s2)
            == seq_fold_left f acc s1 `f` seq_fold_left f e s2)
  (decreases Seq.length s2)
= if Seq.length s2 = 0
  then (
    assert (Seq.equal (append s1 s2) s1)
  )
  else(
    let s2', last = Seq.un_snoc s2 in
    assert (Seq.length s2' < Seq.length s2);
    calc (==) {
      seq_fold_left f acc (append s1 s2);
    (==) { }
      seq_fold_left f acc (append s1 (Seq.snoc s2' last));
    (==) { Seq.append_assoc s1 s2' (Seq.create 1 last) }
      seq_fold_left f acc ((append s1 s2') `append` (Seq.create 1 last));
    (==) { seq_fold_left_append f acc (append s1 s2') (Seq.create 1 last) }
      f (seq_fold_left f acc (append s1 s2')) last;
    (==) { lemma_seq_fold_left_sum' acc e f s1 s2' }
      f (f (seq_fold_left f acc s1) (seq_fold_left f e s2')) last;
    (==) {}
      f (seq_fold_left f acc s1) (f (seq_fold_left f e s2') last);
    (==) {  seq_fold_left_append f e  s2' (Seq.create 1 last) }
      f (seq_fold_left f acc s1) (seq_fold_left f e s2);
    }
  )

let lemma_seq_fold_left_sum
  (#a:Type) (e:a) (f: a -> a -> a)
  (s1 s2 : seq a)
: Lemma
  (requires is_monoid e f)
  (ensures seq_fold_left f e (append s1 s2)
            == seq_fold_left f e s1 `f` seq_fold_left f e s2)
= lemma_seq_fold_left_sum' e e f s1 s2

let lem_append_slice (#a:Type) (s : seq a) (i j k : nat)
  : Lemma (requires i <= j /\ j <= k /\ k <= length s)
          (ensures append (slice s i j) (slice s j k) == slice s i k)
  = assert (Seq.equal (append (slice s i j) (slice s j k)) (slice s i k))

let lem_one_elem (#a:Type) (s : seq a) (v : a)
  : Lemma (requires length s == 1 /\ s @! 0 == v)
          (ensures s == seq![v])
  = assert (Seq.equal s seq![v])

let lemma_upd_index (#a:Type) (s : seq a) (i : nat{i < Seq.length s})
  : Lemma (requires True)
          (ensures (Seq.upd s i (Seq.index s i)) == s)
  = assert (Seq.equal (Seq.upd s i ( Seq.index s i)) s); ()

let lem_seq_refine_at #a (p : a -> prop)
  (s : seq a { forall i. p (s @! i) })
  (i : nat {i < Seq.length s})
  : Lemma ((seq_refine p s) @! i == s @! i)
  = ()
