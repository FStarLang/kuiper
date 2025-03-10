module Kuiper.IsReduction

open Kuiper.Functions
open FStar.Seq

val is_permutation (#a:Type) (s1 s2 : seq a) : prop

noeq
type is_reduction (#a:Type0) (z:a) (f : a -> a -> a) : (s : seq a) -> (r : a) -> Type0 =
  | Emp :
    is_reduction z f seq![] z
  | Singl :
    r:a ->
    is_reduction z f seq![r] r
  | Split :
    s1:seq a -> s2:seq a -> r1:a -> r2:a ->
    is_reduction z f s1 r1 ->
    is_reduction z f s2 r2 ->
    is_reduction z f (s1 `Seq.append` s2) (f r1 r2)
  | Perm :
    s1:seq a -> s2:seq a ->
    r:a ->
    is_permutation s1 s2 ->
    is_reduction z f s1 r ->
    is_reduction z f s2 r

val lemma_Singl (#a:Type0) (z:a) (f : a -> a -> a) (r : a)
  : Lemma (is_reduction z f seq![r] r)
          [SMTPat (is_reduction z f seq![r] r)]

val ac_eq_foldl
  (#a:Type) (z : a) (f : a -> a -> a) (s : seq a) (r : a)
  : Lemma (requires is_comm_semigroup z f /\ is_reduction z f s r)
          (ensures r == Kuiper.Seq.Common.seq_fold_left f z s)

val assoc_uniq_reduction
  (#a:Type) (z:a) (f : a -> a -> a) (xs : seq a) (r1 r2 : a)
: Lemma (requires is_comm_semigroup z f /\ is_reduction z f xs r1 /\ is_reduction z f xs r2)
        (ensures r1 == r2)

val op_is_reduction
  (#a:Type0) (z:a) (f : a -> a -> a)
  (s1 : seq a) (r1 : a)
  (s2 : seq a) (r2 : a)
: Lemma (requires is_reduction z f s1 r1 /\ is_reduction z f s2 r2)
        (ensures is_reduction z f (s1 `Seq.append` s2) (f r1 r2))
        [SMTPat (is_reduction z f (s1 `Seq.append` s2) (f r1 r2))]
