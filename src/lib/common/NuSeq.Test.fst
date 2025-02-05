module NuSeq.Test

open NuSeq

let split_concat (#a:Type) (s : seq a) (i: natle (len s)) =
  let (sl, sr) = split s i in
  assert (s == (sl @ sr))

let insert_remove (#a:Type) (s : seq a) (i : natle (len s)) (v : seq a) =
  assert (s == remove (insert s i v) i (i + len v))

let add_delete (#a:Type) (s : seq a) (i : natle (len s)) (v : a) =
  assert (s == delete (add s i v) i)

let concat_seq0 (#a:Type) (s : seq a) =
  assert (s == (s @ seq0 #a) /\ s == (seq0 #a @ s))

let update_id (#a:Type) (s : seq a) (i : natlt (len s)) =
  assert (s == (s.[i] <- s.[i]))

let push_pop (#a:Type) (s : seq a) (v : a) =
  let s' = push_l v s in
  assert ((pop_l s')._2 == s)

// let push_pop (#a:Type) (s : seq a) (v : a) =
//   assume ((split (seq1 v @ s) 1)._2 == mk_seq (len s + 1 - 1) (fun i -> (mk_seq (len s + 1) (fun j -> if j = 0 then v else s.[j - 1])).[i + 1]));
//   assume (s == mk_seq (len s) (fun i -> s.[i]));
//   // assert (mk_seq (len s + 1 - 1) (fun i -> (mk_seq (len s + 1) (fun j -> if j = 0 then v else s.[j - 1])).[i + 1]) == mk_seq (len s) (fun i -> s.[i]));
//   assert (s == (split (seq1 v @ s) 1)._2 /\ has_type s (seq a))

let sliceA
  (#a:Type)
  (s:seq a)
  (from : nat)
  (to : nat{from <= to /\ to <= len s})
  : seq a
  = let (_, s2) = split s from in
    let (s3, _) = split s2 (to - from) in
    s3
    
let sliceB
  (#a:Type)
  (s:seq a)
  (from : nat)
  (to : nat{from <= to /\ to <= len s})
  : seq a
  = let (s1, s2) = split s to in
    let (_, s3) = split s1 from in
    s3

let test (#a:Type) (s:seq a) (from:nat) (to : nat{from <= to /\ to <= len s}) =
  assert (sliceA s from to == sliceB s from to)

let removeA
  (#a:Type)
  (s:seq a)
  (from : nat)
  (to : nat{from <= to /\ to <= len s})
  : seq a
  = let (s1, _) = split s from in
    let (_, s2) = split s to in
    s1 @ s2

let removeB
  (#a:Type)
  (s:seq a)
  (from : nat)
  (to : nat{from <= to /\ to <= len s})
  : seq a
  = let (s1, s') = split s from in
    let (_, s2) = split s' (to - from) in
    s1 @ s2

let removeC
  (#a:Type)
  (s:seq a)
  (from : nat)
  (to : nat{from <= to /\ to <= len s})
  : seq a
  = let (s', s2) = split s to in
    let (s1, _) = split s' from in
    s1 @ s2

let test2 (#a:Type) (s:seq a) (from:nat) (to : nat{from <= to /\ to <= len s}) =
  admit();
  assert (removeA s from to == removeB s from to);
  assert (removeA s from to == removeC s from to);
  assert (removeB s from to == removeC s from to);
  ()
