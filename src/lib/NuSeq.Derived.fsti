module NuSeq.Derived

open NuSeq.Base

////////
// derived convenience operations

unfold let slice (#a : Type)
  (s : seq a)
  (from : nat)
  (to : natle (len s){from <= to})
  : seq a
  = let (sl, _) = split s to in
    let (_, sm) = split sl from in
    sm

unfold let remove (#a : Type)
  (s : seq a)
  (from : nat)
  (to : natle (len s){from <= to})
  : seq a
  = let (sl, _) = split s from in
    let (_, sr) = split s to in
    sl @ sr

unfold let insert (#a:Type)
  (s : seq a)
  (i : natle (len s))
  (v : seq a)
  : Tot (seq a)
  = let (sl, sr) = split s i in
    sl @ v @ sr

////////
// single element operations

unfold let add (#a:Type)
  (s : seq a)
  (i : natle (len s))
  (v : a)
  : Tot (seq a)
  = insert s i (seq1 v)

unfold let delete (#a:Type)
  (s : seq a)
  (i : natlt (len s))
  : Tot (seq a)
  = remove s i (i + 1)

// Use `s.[i] <- v` instead
unfold let update (#a:Type)
  (s : seq a)
  (i : natlt (len s))
  (v : a)
  : Tot (seq a)
  = insert (delete s i) i (seq1 v)

unfold let op_String_Assignment = update

unfold let push_l (#a:Type)
  (s : seq a)
  (v : a)
  : Tot (seq a)
  = seq1 v @ s

unfold let push_r (#a:Type)
  (s : seq a)
  (v : a)
  : Tot (seq a)
  = s @ seq1 v

unfold let pop_l (#a:Type)
  (s : seq a{len s > 0})
  : Tot (a * seq a)
  = let (v, s') = split s 1 in
    (v.[0], s')

unfold let pop_r (#a:Type)
  (s : seq a{len s > 0})
  : Tot (seq a * a)
  = let (s', v) = split s (len s - 1) in
    (s', v.[0])
