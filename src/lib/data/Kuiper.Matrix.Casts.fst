module Kuiper.Matrix.Casts
#lang-pulse

open Kuiper

open Kuiper.Tensor
module A2 = Kuiper.Array2
open Kuiper.EMatrix
open Kuiper.EMatrix4

// TODO: define this by composing a bijection with the l.imap injection
#push-options "--z3rlimit 20 --retry 3"
let l2_to_l4 (#m #n : nat) (#mm #nn : pos) (l : layout2 (m * mm) (n * nn)) : layout4 m n mm nn = {
  ulen = l.ulen;
  imap =
    Kuiper.Injection.mk_injection
      (fun (i, (j, (k, (o, ())))) ->
        let i : natlt m = i in
        let j : natlt n = j in
        let k : natlt mm = k in
        let o : natlt nn = o in
        let i : natlt (m * mm) = i * mm + k in
        let j : natlt (n * nn) = j * nn + o in
        l.imap.f (i, (j, ())))
      (fun (i1, (j1, (k1, (o1, ()))))
           (i2, (j2, (k2, (o2, ())))) ->
           l.imap.is_inj (i1 * mm + k1, (j1 * nn + o1, ())) (i2 * mm + k2, (j2 * nn + o2, ()));
           ())
}
#pop-options

let em2_to_em4 (#et : Type0) (#m #n #mm #nn : nat) (e : ematrix et (m * mm) (n * nn)) : ematrix4 et m n mm nn =
  Kuiper.EMatrix4.mkM (fun i j k o -> Kuiper.EMatrix.macc e (i * mm + k) (j * nn + o))

#push-options "--z3rlimit 20 --retry 3"
let l4_to_l2 (#m #n : nat) (#mm #nn : pos) (l : layout4 m n mm nn) : layout2 (m * mm) (n * nn) = {
  ulen = l.ulen;
  imap =
    Kuiper.Injection.mk_injection #(natlt (m * mm) & (natlt (n * nn) & unit))
      (fun (x, (y, ())) ->
        let i : natlt m = x / mm in
        let j : natlt n = y / nn in
        let k : natlt mm = x % mm in
        let o : natlt nn = y % nn in
        l.imap.f (i, (j, (k, (o, ())))))
      (fun (x1, (y1, ()))
           (x2, (y2, ())) ->
           let i1 : natlt m = x1 / mm in
           let j1 : natlt n = y1 / nn in
           let k1 : natlt mm = x1 % mm in
           let o1 : natlt nn = y1 % nn in
           let i2 : natlt m = x2 / mm in
           let j2 : natlt n = y2 / nn in
           let k2 : natlt mm = x2 % mm in
           let o2 : natlt nn = y2 % nn in
           l.imap.is_inj (i1, (j1, (k1, (o1, ())))) (i2, (j2, (k2, (o2, ()))));
           ())
}
#pop-options

let em4_to_em2 (#et : Type0) (#m #n #mm #nn : nat) (e : ematrix4 et m n mm nn) : ematrix et (m * mm) (n * nn) =
  Kuiper.EMatrix.mkM (fun x y ->
    let i : natlt m = x / mm in
    let j : natlt n = y / nn in
    let k : natlt mm = x % mm in
    let o : natlt nn = y % nn in
    Kuiper.EMatrix4.macc e i j k o)

inline_for_extraction noextract
fn m2_to_m4
  (m n mm nn : erased nat{mm > 0 /\ nn > 0})
  (#et : Type0) {| scalar et |}
  (#lA : A2.full_layout (m * mm) (n * nn))
  (gA : A2.t et lA)
  (#eA : ematrix et _ _)
  (#f : perm)
  requires
    gA |-> Frac f eA
  returns
    gA4 : array4 et (l2_to_l4 lA)
  ensures
    gA4 |-> Frac f (em2_to_em4 eA) **
    pure (core gA4 == A2.core gA)
{
  (* Very roundabout way to do this. *)
  A2.lower gA;
  tensor_abs' (l2_to_l4 lA) (A2.core gA);
  let r = from_array (l2_to_l4 lA) (A2.core gA);
  assert rewrites_to r (from_array (l2_to_l4 lA) (A2.core gA));
  assert pure (Kuiper.EMatrix4.equal
    (from_seq (l2_to_l4 lA) (to_seq lA eA))
    (em2_to_em4 eA));
  rewrite r |-> Frac f (from_seq (l2_to_l4 lA) (to_seq lA eA))
       as r |-> Frac f (em2_to_em4 eA);
  ();
  r
}

inline_for_extraction noextract
fn m4_to_m2
  (m n mm nn : erased nat{mm > 0 /\ nn > 0})
  (#et : Type0) {| scalar et |}
  (#lA4 : full_layout4 m n mm nn)
  (gA4 : array4 et lA4)
  (#eA4 : ematrix4 et _ _ _ _)
  (#f : perm)
  requires
    gA4 |-> Frac f eA4
  returns
    gA : A2.t et (l4_to_l2 lA4)
  ensures
    gA |-> Frac f (em4_to_em2 eA4) **
    pure (core gA4 == A2.core gA)
{
  (* Very roundabout way to do this. *)
  tensor_concr gA4;
  A2.raise' (l4_to_l2 lA4) (core gA4);
  let r = A2.from_array (l4_to_l2 lA4) (core gA4);
  assert rewrites_to r (A2.from_array (l4_to_l2 lA4) (core gA4));
  assert pure (Kuiper.EMatrix.equal
    (from_seq (l4_to_l2 lA4) (to_seq lA4 eA4))
    (em4_to_em2 eA4));
  rewrite r |-> Frac f (from_seq (l4_to_l2 lA4) (to_seq lA4 eA4))
       as r |-> Frac f (em4_to_em2 eA4);
  ();
  r
}
