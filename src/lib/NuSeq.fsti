module NuSeq

[@@erasable]
new
val seq (a:Type u#aa) : Type u#aa

type natlt (b:int) = n:nat{n <  b}
type natle (b:int) = n:nat{n <= b}

val len
  (#a:Type)
  (s:seq a)
  : nat

// unfold let ( !! ) = len

val empty
  (#a:Type)
  : seq a

val empty_len
  (a:Type)
  : Lemma (len (empty #a) == 0)
          [SMTPat (len (empty #a))]

val index
  (#a : Type)
  (s : seq a)
  (i : natlt (len s))
  : Tot a

val concat
  (#a:Type)
  (s1 s2 : seq a)
  : seq a

unfold let (@) = concat
unfold let op_String_Access (#a:Type) (s:seq a) (i : natlt (len s)) = index s i
// let op_String_Assignment (x:'a) (l:lens 'a 'b) (v:'b) : 'a = (x |:= l) v

val concat_len
  (#a:Type) (s1 s2 : seq a)
  : Lemma (len s1 + len s2 == len (s1 @ s2))
          [SMTPat (len (s1 @ s2))]

val concat_index_l
  (#a:Type) (s1 s2 : seq a) (i:natlt (len s1 + len s2))
  : Lemma (requires i < len s1)
          (ensures (s1 @ s2).[i] == s1.[i])
          [SMTPat (s1 @ s2).[i]]

val concat_index_r
  (#a:Type) (s1 s2 : seq a) (i:natlt (len s1 + len s2))
  : Lemma (requires i >= len s1)
          (ensures (s1 @ s2).[i] == s2.[i - len s1])
          [SMTPat (s1 @ s2).[i]]

val split
  (#a:Type)
  (s : seq a)
  (i : natle (len s))
  : Tot (seq a * seq a)

val split_len_l
  (#a:Type)
  (s : seq a)
  (i : natle (len s))
  : Lemma (len (fst (split s i)) == i)
          [SMTPat (split s i)]

val split_len_r
  (#a:Type)
  (s : seq a)
  (i : natle (len s))
  : Lemma (len (snd (split s i)) == len s - i)
          [SMTPat (split s i)]

let match_between
  (#a:Type)
  (lo hi : int)
  (shift : int)
  (s1 s2 : seq a)
  : prop
  = forall j. 0 <= j /\ j < len s1 /\
              lo <= j /\ j < hi /\
              0 <= shift + j /\ j + shift < len s2
              ==> s1.[j] == s2.[shift+j]

let equal (#a:Type) (s1 s2 : seq a) =
  len s1 == len s2 /\
  match_between 0 (len s1) 0 s1 s2

val split_index_l
  (#a:Type)
  (s : seq a)
  (i : natle (len s))
  : Lemma (match_between 0 i 0 (fst (split s i)) s)
          [SMTPat (split s i)]

val split_index_r
  (#a:Type)
  (s : seq a)
  (i : natle (len s))
  : Lemma (match_between 0 (len s - i) i (snd (split s i)) s)
          [SMTPat (split s i)]
