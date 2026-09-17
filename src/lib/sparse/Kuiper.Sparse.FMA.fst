module Kuiper.Sparse.FMA

#lang-pulse
open Kuiper
open Kuiper.Seq.Common { seq_replace }

inline_for_extraction noextract
let fma (#et : Type0) {| scalar et |} (y x1 x2 : et) : et = y `add` (x1 `mul` x2)

noextract
let seq_fma
  (#et : Type0) {| scalar et |}
  (x1 : et)
  (#n : nat)
  (x2 : lseq et n)
  (#sz_y : nat)
  (y : lseq et sz_y)
  (k : nat { k + n <= sz_y })
  (to : natle n)
: lseq et sz_y
= seq_replace y k (k + to) (Seq.init to fun i -> fma (y @! k + i) x1 (x2 @! i))

inline_for_extraction noextract
fn fma_arr
  (#et : Type0) {| scalar et |}
  (x1 : et)
  (n : sz)
  (x2 : larray et n)
  (#vx2 : erased (lseq et n))
  (#sz_y : erased nat)
  (y : larray et sz_y)
  (#vy : erased (lseq et sz_y))
  (k : sz { k + n <= sz_y })
  preserves gpu
  preserves x2 |-> vx2
  requires  y |-> vy
  ensures  y |-> seq_fma x1 vx2 vy k n
{
  let mut ix : sz = 0sz;
  while (!ix <^ n)
    invariant exists* vix (vy' : lseq et sz_y).
      ix |-> vix **
      y |-> vy' **
      pure (
        vix <= n /\
        (forall (i : natlt sz_y).
          vy' @! i ==
            (if k <= i && i < k + vix
             then fma (vy @! i) x1 (vx2 @! (i - k))
             else vy @! i))
      )
    decreases (n - !ix)
  {
    open Pulse.Lib.Array;
    y.(k +^ !ix) <- fma y.(k +^ !ix) x1 x2.(!ix);
    ix := !ix +^ 1sz;
  };

  with vy'. assert y |-> vy';
  assert pure (
    Seq.equal vy' (seq_fma x1 vx2 vy k n)
  );
}

noextract
let seq_fma'
  (#et : Type0) {| scalar et |}
  (cnt : nat)
  (x1 : et)
  (#n : nat { cnt /? n })
  (x2 : lseq et n)
  (#sz_y : nat)
  (y : lseq et sz_y)
  (k1 : nat { k1 + cnt <= sz_y })
  (k2 : nat { cnt /? k2 })
: Tot (lseq et sz_y)
=
  if k2 < n
    then seq_fma x1 #cnt (Seq.slice x2 k2 (k2 + cnt)) y k1 cnt
    else y