module GPU.Seq.Common

open FStar.Seq

unfold
let ( @! ) (#a:Type) (s : seq a) (i : nat { i < Seq.length s }) : a = Seq.index #a s i

let rec seq_fold_left (#t:Type) (f: t -> t -> t) (acc: t) (v: seq t)
: Tot t (decreases length v)
=
  if length v = 0 then
    acc
  else
    let hd = head v in
    let tl = tail v in
    seq_fold_left f (f acc hd) tl
