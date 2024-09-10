module GPU.Seq.Common

open FStar.Seq

let lemma_seq_fold_left_sum (#a:Type) (e:a) (f: a -> a -> a)
  (s1 s2 : seq a)
  : Lemma (requires is_monoid e f)
          (ensures seq_fold_left f e (append s1 s2)
                   == f (seq_fold_left f e s1) (seq_fold_left f e s2))
  = admit() (* boring *)

let lem_append_slice (#a:Type) (s : seq a) (i j k : nat)
  : Lemma (requires i <= j /\ j <= k /\ k <= length s)
          (ensures append (slice s i j) (slice s j k) == slice s i k)
  = assert (Seq.equal (append (slice s i j) (slice s j k)) (slice s i k))
