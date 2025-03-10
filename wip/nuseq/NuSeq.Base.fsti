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

val mk_seq (#a:Type)
  (len: nat)
  (f : natlt len -> a): seq a

val mk_seq_ghost (#a:Type)
  (len: nat)
  (f : natlt len -> GTot a): GTot (seq a)

////////
// len lemmas

val mk_seq_len (#a:Type) (l: nat) (f : natlt l -> a)
  : Lemma (len (mk_seq l f) == l)
  [SMTPat (len (mk_seq l f))]

val mk_seq_ghost_len (#a:Type) (l: nat) (f : natlt l -> GTot a)
  : Lemma (len (mk_seq_ghost l f) == l)
  [SMTPat (len (mk_seq_ghost l f))]

////////
// index lemmas

val mk_seq_index (#a:Type) (l: nat) (f : natlt l -> a) (i : natlt l)
  : Lemma ((mk_seq l f).[i] == f i)
  [SMTPat  (mk_seq l f).[i]]

val mk_seq_ghost_index (#a:Type) (l: nat) (f : natlt l -> GTot a) (i : natlt l)
  : Lemma ((mk_seq_ghost l f).[i] == f i)
  [SMTPat  (mk_seq_ghost l f).[i]]

////////
// equality lemma

val equal_defn_0 (#a:Type) (len : nat) (f1 f2 : natlt len -> a)
  : Lemma (requires (forall (i: int). 0 <= i /\ i < len ==> f1 i == f2 i))
          (ensures mk_seq len f1 == mk_seq len f2)
          [SMTPat (mk_seq len f1); SMTPat (mk_seq len f2)]

val equal_defn_1 (#a:Type) (len : nat) (f1 f2 : natlt len -> GTot a)
  : Lemma (requires (forall (i: int). 0 <= i /\ i < len ==> f1 i == f2 i))
          (ensures mk_seq_ghost len f1 == mk_seq_ghost len f2)
          [SMTPat (mk_seq_ghost len f1); SMTPat (mk_seq_ghost len f2)]

val equal_defn_2 (#a:Type) (len : nat) (f1 : natlt len -> a) (f2 : natlt len -> GTot a)
  : Lemma (requires (forall (i: int). 0 <= i /\ i < len ==> f1 i == f2 i))
          (ensures mk_seq len f1 == mk_seq_ghost len f2)
          [SMTPat (mk_seq len f1); SMTPat (mk_seq_ghost len f2)]

#push-options "--warn_error -288"
val as_mk_seq (#a:Type) (s : seq a)
  : Lemma (s == mk_seq (len s) (fun i -> s.[i]))
  [SMTPat (has_type s (seq a))] // OK? Useful?
#pop-options
