module NuSeq.Derived

open NuSeq.Base

////////
// primitive fixed size

let seq0 (#a:Type)
  : seq a
  = mk_seq 0 (fun i -> match i with)

let seq1 (#a:Type)
  (v0 : a): seq a
  = mk_seq 1 (fun i -> match i with | 0 -> v0)

let seq2 (#a:Type)
  (v0 v1 : a): seq a
  = mk_seq 2 (fun i -> match i with | 0 -> v0 | 1 -> v1)

let seq3 (#a:Type)
  (v0 v1 v2 : a): seq a
  = mk_seq 3 (fun i -> match i with | 0 -> v0 | 1 -> v1 | 2 -> v2)

let seq4 (#a:Type)
  (v0 v1 v2 v3 : a): seq a
  = mk_seq 4 (fun i -> match i with | 0 -> v0 | 1 -> v1 | 2 -> v2 | 3 -> v3)

let seq5 (#a:Type)
  (v0 v1 v2 v3 v4 : a): seq a
  = mk_seq 5 (fun i -> match i with | 0 -> v0 | 1 -> v1 | 2 -> v2 | 3 -> v3 | 4 -> v4)

let seq6 (#a:Type)
  (v0 v1 v2 v3 v4 v5 : a): seq a
  = mk_seq 6 (fun i -> match i with | 0 -> v0 | 1 -> v1 | 2 -> v2 | 3 -> v3 | 4 -> v4 | 5 -> v5)

let seq7 (#a:Type)
  (v0 v1 v2 v3 v4 v5 v6 : a): seq a
  = mk_seq 7 (fun i -> match i with | 0 -> v0 | 1 -> v1 | 2 -> v2 | 3 -> v3 | 4 -> v4 | 5 -> v5 | 6 -> v6)

let seq8 (#a:Type)
  (v0 v1 v2 v3 v4 v5 v6 v7 : a): seq a
  = mk_seq 8 (fun i -> match i with | 0 -> v0 | 1 -> v1 | 2 -> v2 | 3 -> v3 | 4 -> v4 | 5 -> v5 | 6 -> v6 | 7 -> v7)


////////
// derived convenience operations

// Use `s1 @ s2` instead
let concat (#a:Type)
  (s1 s2 : seq a)
  : seq a
  = mk_seq (len s1 + len s2) (fun i -> if i < len s1 then s1.[i] else s2.[i - len s1])

unfold let (@) = concat

let split (#a:Type)
  (s : seq a)
  (i : natle (len s))
  : Tot (seq a * seq a)
  = (mk_seq i (fun j -> s.[j]), mk_seq (len s - i) (fun j -> s.[i + j]))

unfold let slice (#a : Type)
  (s : seq a)
  (from : nat)
  (to : natle (len s){from <= to})
  : seq a
  = mk_seq (to - from) (fun i -> s.[from + i])

unfold let remove (#a : Type)
  (s : seq a)
  (from : nat)
  (to : natle (len s){from <= to})
  : seq a
  = mk_seq (len s - (to - from)) (fun i -> if i < from then s.[i] else s.[to + i - from])

unfold let insert (#a:Type)
  (s : seq a)
  (i : natle (len s))
  (v : seq a)
  : Tot (seq a)
  = mk_seq (len s + len v) (fun j -> if j < i then s.[j] else if j < i + len v then v.[j - i] else s.[j - len v])

unfold let zip (#a #b:Type)
  (s1 : seq a)
  (s2 : seq b{len s1 == len s2})
  : seq (a * b)
  = mk_seq (len s1) (fun i -> (s1.[i], s2.[i]))

unfold let unzip (#a #b:Type)
  (s : seq (a * b))
  : seq a * seq b
  = (mk_seq (len s) (fun i -> (s.[i])._1), mk_seq (len s) (fun i -> (s.[i])._2))

unfold let map (#a #b:Type)
  (s : seq a)
  (f : a -> b)
  : seq b
  = mk_seq (len s) (fun i -> f s.[i])

////////
// single element operations

unfold let add (#a:Type)
  (s : seq a)
  (i : natle (len s))
  (v : a)
  : Tot (seq a)
  = mk_seq (len s + 1) (fun j -> if j < i then s.[j] else if j = i then v else s.[j - 1])

unfold let delete (#a:Type)
  (s : seq a)
  (i : natlt (len s))
  : Tot (seq a)
  = mk_seq (len s - 1) (fun j -> if j < i then s.[j] else s.[j + 1])

// Use `s.[i] <- v` instead
unfold let update (#a:Type)
  (s : seq a)
  (i : natlt (len s))
  (v : a)
  : Tot (seq a)
  = mk_seq (len s) (fun j -> if j = i then v else s.[j])

unfold let op_String_Assignment = update

unfold let push_l (#a:Type)
  (v : a)
  (s : seq a)
  : Tot (seq a)
  = mk_seq (len s + 1) (fun i -> if i = 0 then v else s.[i - 1])

unfold let push_r (#a:Type)
  (s : seq a)
  (v : a)
  : Tot (seq a)
  = mk_seq (len s + 1) (fun i -> if i = len s then v else s.[i])

unfold let pop_l (#a:Type)
  (s : seq a{len s > 0})
  : Tot (a * seq a)
  = let (_, s') = split s 1 in
    (s.[0], s')

unfold let pop_r (#a:Type)
  (s : seq a{len s > 0})
  : Tot (seq a * a)
  = let (s', _) = split s (len s - 1) in
    (s', s.[len s - 1])

////////
// reduction operations

let rec fold_l (#a #b : Type) (s : seq a) (init : b) (f : a -> b -> b) : Tot b (decreases len s) =
  if len s = 0 then init else (let (hd, tl) = pop_l s in fold_l tl (f hd init) f)

let rec fold_r (#a : Type) (s : seq a) (init : a) (f : a -> a -> a) : Tot a (decreases len s) =
  if len s = 0 then init else (let (hd, tl) = pop_l s in f hd (fold_r tl init f))
