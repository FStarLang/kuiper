module Kuiper.Sparse.FMA

#lang-pulse
open Kuiper
open Kuiper.Seq.Common { seq_replace }

inline_for_extraction noextract
let fma (#et : Type0) {| scalar et |} (y x1 x2 : et) : et = y `add` (x1 `mul` x2)

let seq_fma
  (x1 : real)
  (#n : nat)
  (x2 : lseq real n)
  (#sz_y : nat)
  (y : lseq real sz_y)
  (k : nat { k + n <= sz_y })
  (to : natle n)
: GTot (lseq real sz_y)
= seq_replace y k (k + to) (Seq.init to fun i -> fma (y @! k + i) x1 (x2 @! i))

inline_for_extraction noextract
fn fma_arr
  (#et : Type0) {| scalar et, real_like et |}
  (x1 : et)
  (n : sz)
  (x2 : larray et n)
  (#vx2 : erased (lseq et n))
  (#sz_y : erased nat)
  (y : larray et sz_y)
  (#vy : erased (lseq et sz_y))
  (#rx1 : erased real)
  (#rx2 : erased (lseq real n))
  (#ry : erased (lseq real sz_y))
  (k : sz { k + n <= sz_y })
  preserves gpu
  preserves x2 |-> vx2
  requires  y |-> vy
  requires pure (x1 %~ rx1 /\ vx2 %~ rx2 /\ vy %~ ry)
  ensures exists* (vy' : lseq et sz_y).
    y |-> vy' ** pure (vy' %~ seq_fma (reveal rx1) (reveal rx2) (reveal ry) k n)
{
  let mut ix : sz = 0sz;
  while (!ix <^ n)
    invariant exists* vix (vy' : lseq et sz_y).
      ix |-> vix **
      y |-> vy' **
      pure (
        vix <= n /\
        (forall (i : natlt sz_y).
          (vy' @! i) %~
            (if k <= i && i < k + vix
             then fma (ry @! i) (reveal rx1) (rx2 @! (i - k))
             else ry @! i))
      )
    decreases (n - !ix)
  {
    open Pulse.Lib.Array;
    y.(k +^ !ix) <- fma y.(k +^ !ix) x1 x2.(!ix);
    ix := !ix +^ 1sz;
  };

  with vy'. assert y |-> vy';
  assert pure (
    vy' %~ seq_fma (reveal rx1) (reveal rx2) (reveal ry) k n
  );
}

(* No SMTPat: triggering on divides interferes with the termination proof
   in seq_fma'. *)
let __divides_next (cnt k n : nat)
  : Lemma (requires cnt /? k /\ cnt /? n /\ k < n)
          (ensures k + cnt <= n)
  = if cnt = 0 then ()
    else begin
      lemma_divides_exact cnt k;
      lemma_divides_exact cnt n;
      FStar.Math.Lemmas.lemma_mult_le_left cnt (k / cnt + 1) (n / cnt)
    end

let seq_fma'
  (cnt : nat)
  (x1 : real)
  (#n : nat { cnt /? n })
  (x2 : lseq real n)
  (#sz_y : nat)
  (y : lseq real sz_y)
  (k1 : nat { k1 + cnt <= sz_y })
  (k2 : nat { cnt /? k2 })
: GTot (lseq real sz_y)
=
  if k2 < n
    then (__divides_next cnt k2 n;
          seq_fma x1 #cnt (Seq.slice x2 k2 (k2 + cnt)) y k1 cnt)
    else y
