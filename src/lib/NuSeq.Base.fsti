module NuSeq.Base

[@@erasable]
new
val seq (a:Type u#aa) : Type u#aa

type natlt (b:int) = n:nat{n <  b}
type natle (b:int) = n:nat{n <= b}

////////
// primitive specification fns

val len (#a:Type)
  (s:seq a)
  : nat

// Use `s.[i]` instead (which desugars to `op_String_Access`)
val index (#a : Type)
  (s : seq a)
  (i : natlt (len s))
  : Tot a

unfold let op_String_Access = index
// let op_String_Assignment (x:'a) (l:lens 'a 'b) (v:'b) : 'a = (x |:= l) v

// Use `s1 == s2` instead
val equal (#a : Type)
  (s1 s2 : seq a)
  : prop

// TODO: would be nice to make this automatic, `[SMTPat (s1 == s2)]` doesn't work
// Note: the `equal` precondition will be proved automatically
val assert_equal (#a:Type) (s1 s2 : seq a)
  : Lemma (requires equal s1 s2)
          (ensures s1 == s2)

////////
// primitive operations

// Use `s1 @ s2` instead
val concat (#a:Type)
  (s1 s2 : seq a)
  : seq a

unfold let (@) = concat

val split (#a:Type)
  (s : seq a)
  (i : natle (len s))
  : Tot (seq a * seq a)

////////
// primitive fixed size

val seq0 (#a:Type)
  : seq a

val seq1 (#a:Type)
  (v0 : a): seq a

val seq2 (#a:Type)
  (v0 v1 : a): seq a

val seq3 (#a:Type)
  (v0 v1 v2 : a): seq a

val seq4 (#a:Type)
  (v0 v1 v2 v3 : a): seq a

val seq5 (#a:Type)
  (v0 v1 v2 v3 v4 : a): seq a

val seq6 (#a:Type)
  (v0 v1 v2 v3 v4 v5 : a): seq a

val seq7 (#a:Type)
  (v0 v1 v2 v3 v4 v5 v6 : a): seq a

val seq8 (#a:Type)
  (v0 v1 v2 v3 v4 v5 v6 v7 : a): seq a





////////
// equality lemma

val equal_defn (#a:Type) (s1 s2 : seq a)
  : Lemma (requires len s1 == len s2 /\ forall (j: int). 0 <= j /\ j < len s1 ==> s1.[j] == s2.[j])
          (ensures equal s1 s2)
          [SMTPat (equal s1 s2)]

////////
// len lemmas

val concat_len (#a:Type) (s1 s2 : seq a)
  : Lemma (len (s1 @ s2) == len s1 + len s2)
  [SMTPat (len (s1 @ s2))]

val split_len_l (#a:Type)
  (s : seq a) (i : natle (len s))
  : Lemma (len (fst (split s i)) == i)
          [SMTPat (split s i)]

val split_len_r (#a:Type)
  (s : seq a) (i : natle (len s))
  : Lemma (len (snd (split s i)) == len s - i)
          [SMTPat (split s i)]

val seq0_len (#a:Type)
  : Lemma (len (seq0 #a) == 0)
  [SMTPat (len (seq0 #a))]

val seq1_len (#a:Type) (v0 : a)
  : Lemma (len (seq1 v0) == 1)
  [SMTPat (len (seq1 v0))]

val seq2_len (#a:Type) (v0 v1 : a)
  : Lemma (len (seq2 v0 v1) == 2)
  [SMTPat (len (seq2 v0 v1))]

val seq3_len (#a:Type) (v0 v1 v2 : a)
  : Lemma (len (seq3 v0 v1 v2) == 3)
  [SMTPat (len (seq3 v0 v1 v2))]

val seq4_len (#a:Type) (v0 v1 v2 v3 : a)
  : Lemma (len (seq4 v0 v1 v2 v3) == 4)
  [SMTPat (len (seq4 v0 v1 v2 v3))]

val seq5_len (#a:Type) (v0 v1 v2 v3 v4 : a)
  : Lemma (len (seq5 v0 v1 v2 v3 v4) == 5)
  [SMTPat (len (seq5 v0 v1 v2 v3 v4))]

val seq6_len (#a:Type) (v0 v1 v2 v3 v4 v5 : a)
  : Lemma (len (seq6 v0 v1 v2 v3 v4 v5) == 6)
  [SMTPat (len (seq6 v0 v1 v2 v3 v4 v5))]

val seq7_len (#a:Type) (v0 v1 v2 v3 v4 v5 v6 : a)
  : Lemma (len (seq7 v0 v1 v2 v3 v4 v5 v6) == 7)
  [SMTPat (len (seq7 v0 v1 v2 v3 v4 v5 v6))]

val seq8_len (#a:Type) (v0 v1 v2 v3 v4 v5 v6 v7 : a)
  : Lemma (len (seq8 v0 v1 v2 v3 v4 v5 v6 v7) == 8)
  [SMTPat (len (seq8 v0 v1 v2 v3 v4 v5 v6 v7))]

////////
// index lemmas

// TODO: which is better?

val concat_index_l (#a:Type)
  (s1 s2 : seq a) (i : natlt (len s1 + len s2))
  : Lemma (requires i < len s1)
          (ensures (s1 @ s2).[i] == s1.[i])
          [SMTPat (s1 @ s2).[i]]

val concat_index_r (#a:Type)
  (s1 s2 : seq a) (i : natlt (len s1 + len s2))
  : Lemma (requires i >= len s1)
          (ensures (s1 @ s2).[i] == s2.[i - len s1])
          [SMTPat (s1 @ s2).[i]]

// val concat_index (#a:Type)
//   (s1 s2 : seq a) (i : natlt (len s1 + len s2))
//   : Lemma ((s1 @ s2).[i] == (if i < len s1 then s1.[i] else s2.[i - len s1]))
//           [SMTPat (s1 @ s2).[i]]

val split_index_l (#a:Type)
  (s : seq a) (i : natle (len s)) (j : natlt i)
  : Lemma ((fst (split s i)).[j] == s.[j])
  [SMTPat  (fst (split s i)).[j]]

val split_index_r (#a:Type)
  (s : seq a) (i : natle (len s)) (j : natlt (len s - i))
  : Lemma ((snd (split s i)).[j] == s.[i + j])
  [SMTPat  (snd (split s i)).[j]]

val seq1_index (#a:Type) (v0 : a) (i : natlt 1)
  : Lemma ((seq1 v0).[i] == v0)
  [SMTPat  (seq1 v0).[i]]

val seq2_index (#a:Type) (v0 v1 : a) (i : natlt 2)
  : Lemma ((seq2 v0 v1).[i] == (match i with | 0 -> v0 | 1 -> v1))
  [SMTPat  (seq2 v0 v1).[i]]

val seq3_index (#a:Type) (v0 v1 v2 : a) (i : natlt 3)
  : Lemma ((seq3 v0 v1 v2).[i] == (match i with | 0 -> v0 | 1 -> v1 | 2 -> v2))
  [SMTPat  (seq3 v0 v1 v2).[i]]

val seq4_index (#a:Type) (v0 v1 v2 v3 : a) (i : natlt 4)
  : Lemma ((seq4 v0 v1 v2 v3).[i] == (match i with | 0 -> v0 | 1 -> v1 | 2 -> v2 | 3 -> v3))
  [SMTPat  (seq4 v0 v1 v2 v3).[i]]

val seq5_index (#a:Type) (v0 v1 v2 v3 v4 : a) (i : natlt 5)
  : Lemma ((seq5 v0 v1 v2 v3 v4).[i] == (match i with | 0 -> v0 | 1 -> v1 | 2 -> v2 | 3 -> v3 | 4 -> v4))
  [SMTPat  (seq5 v0 v1 v2 v3 v4).[i]]

val seq6_index (#a:Type) (v0 v1 v2 v3 v4 v5 : a) (i : natlt 6)
  : Lemma ((seq6 v0 v1 v2 v3 v4 v5).[i] == (match i with | 0 -> v0 | 1 -> v1 | 2 -> v2 | 3 -> v3 | 4 -> v4 | 5 -> v5))
  [SMTPat  (seq6 v0 v1 v2 v3 v4 v5).[i]]

val seq7_index (#a:Type) (v0 v1 v2 v3 v4 v5 v6 : a) (i : natlt 7)
  : Lemma ((seq7 v0 v1 v2 v3 v4 v5 v6).[i] == (match i with | 0 -> v0 | 1 -> v1 | 2 -> v2 | 3 -> v3 | 4 -> v4 | 5 -> v5 | 6 -> v6))
  [SMTPat  (seq7 v0 v1 v2 v3 v4 v5 v6).[i]]

val seq8_index (#a:Type) (v0 v1 v2 v3 v4 v5 v6 v7 : a) (i : natlt 8)
  : Lemma ((seq8 v0 v1 v2 v3 v4 v5 v6 v7).[i] == (match i with | 0 -> v0 | 1 -> v1 | 2 -> v2 | 3 -> v3 | 4 -> v4 | 5 -> v5 | 6 -> v6 | 7 -> v7))
  [SMTPat  (seq8 v0 v1 v2 v3 v4 v5 v6 v7).[i]]
