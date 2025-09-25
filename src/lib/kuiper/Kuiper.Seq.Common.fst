module Kuiper.Seq.Common

open FStar.Seq

let lemma_seq_fold_monoid
  (#a:Type) (e:a) (f: a -> a -> a)
  (s : seq a)
  : Lemma (requires is_monoid e f)
          (ensures seq_fold_left f e s == seq_fold_right f s e)
  = admit()

let lemma_seq_fold_right_sum
  (#a:Type) (e:a) (f: a -> a -> a)
  (s1 s2 : seq a)
  : Lemma (requires is_monoid e f)
          (ensures seq_fold_right f (append s1 s2) e
                   == seq_fold_right f s1 e `f` seq_fold_right f s2 e)
  = admit()

let lemma_seq_fold_left_sum
  (#a:Type) (e:a) (f: a -> a -> a)
  (s1 s2 : seq a)
  : Lemma (requires is_monoid e f)
          (ensures seq_fold_left f e (append s1 s2)
                   == seq_fold_left f e s1 `f` seq_fold_left f e s2)
  = lemma_seq_fold_monoid e f s1;
    lemma_seq_fold_monoid e f s2;
    lemma_seq_fold_monoid e f (append s1 s2);
    lemma_seq_fold_right_sum e f s1 s2;
    ()

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
