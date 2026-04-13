module Kuiper.Matrix.Casts
#lang-pulse

open Kuiper

module EM = Kuiper.EMatrix
module A2 = Kuiper.Array2
module A4 = Kuiper.Array4
open Kuiper.EMatrix
open Kuiper.EMatrix4

let l2_to_l4 (#m #n : nat) (#mm #nn : pos) (l : A2.layout (m * mm) (n * nn)) : A4.layout m n mm nn = {
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
           ())
}

let em2_to_em4 (#et : Type0) (#m #n #mm #nn : nat) (e : ematrix et (m * mm) (n * nn)) : ematrix4 et m n mm nn =
  Kuiper.EMatrix4.mkM (fun i j k o -> Kuiper.EMatrix.macc e (i * mm + k) (j * nn + o))

let l4_to_l2 (#m #n : nat) (#mm #nn : pos) (l : A4.layout m n mm nn) : A2.layout (m * mm) (n * nn) = {
  ulen = l.ulen;
  imap =
    Kuiper.Injection.mk_injection #(natlt (m * mm) & (natlt (n * nn) & unit))
      (fun (i, (j, ())) ->
        let i : natlt m = i / mm in
        let j : natlt n = j / nn in
        let k : natlt mm = i % mm in
        let o : natlt nn = j % nn in
        l.imap.f (i, (j, (k, (o, ())))))
      (fun (i1, (j1, ()))
           (i2, (j2, ())) ->
           admit())
}

let em4_to_em2 (#et : Type0) (#m #n #mm #nn : nat) (e : ematrix4 et m n mm nn) : ematrix et (m * mm) (n * nn) =
  Kuiper.EMatrix.mkM (fun i j ->
    let i : natlt m = i / mm in
    let j : natlt n = j / nn in
    let k : natlt mm = i % mm in
    let o : natlt nn = j % nn in
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
    gA4 : A4.t et (l2_to_l4 lA)
  ensures
    gA4 |-> Frac f (em2_to_em4 eA) **
    pure (A4.core gA4 == A2.core gA)
{
  (* Very roundabout way to do this. *)
  A2.lower gA;
  A4.raise' (l2_to_l4 lA) (A2.core gA);
  let r = A4.from_array (l2_to_l4 lA) (A2.core gA);
  assert rewrites_to r (A4.from_array (l2_to_l4 lA) (A2.core gA));
  assert pure (Kuiper.EMatrix4.equal
    (A4.from_seq (l2_to_l4 lA) (A2.to_seq lA eA))
    (em2_to_em4 eA));
  rewrite r |-> Frac f (A4.from_seq (l2_to_l4 lA) (A2.to_seq lA eA))
       as r |-> Frac f (em2_to_em4 eA);
  ();
  r
}

inline_for_extraction noextract
fn m4_to_m2
  (m n mm nn : erased nat{mm > 0 /\ nn > 0})
  (#et : Type0) {| scalar et |}
  (#lA4 : A4.full_layout m n mm nn)
  (gA4 : A4.t et lA4)
  (#eA4 : ematrix4 et _ _ _ _)
  (#f : perm)
  requires
    gA4 |-> Frac f eA4
  returns
    gA : A2.t et (l4_to_l2 lA4)
  ensures
    gA |-> Frac f (em4_to_em2 eA4) **
    pure (A4.core gA4 == A2.core gA)
{
  (* Very roundabout way to do this. *)
  A4.lower gA4;
  A2.raise' (l4_to_l2 lA4) (A4.core gA4);
  let r = A2.from_array (l4_to_l2 lA4) (A4.core gA4);
  assert rewrites_to r (A2.from_array (l4_to_l2 lA4) (A4.core gA4));
  assert pure (Kuiper.EMatrix.equal
    (A2.from_seq (l4_to_l2 lA4) (A4.to_seq lA4 eA4))
    (em4_to_em2 eA4));
  rewrite r |-> Frac f (A2.from_seq (l4_to_l2 lA4) (A4.to_seq lA4 eA4))
       as r |-> Frac f (em4_to_em2 eA4);
  ();
  r
}
