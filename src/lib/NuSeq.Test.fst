module NuSeq.Test

open NuSeq

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
  assert (sliceA s from to `equal` sliceB s from to)

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
  assert (removeA s from to `equal` removeB s from to);
  assert (removeA s from to `equal` removeC s from to);
  assert (removeB s from to `equal` removeC s from to);
  ()

