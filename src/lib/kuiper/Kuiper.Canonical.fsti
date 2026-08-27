module Kuiper.Canonical

(* [canonical x] pins [x] as *the* distinguished inhabitant of its type, so
that two values known canonical are equal. Refining a superclass field with
it (see [Kuiper.Scalars.Base.scalar]) makes an instance reached by two
different resolution paths provably the same instance. *)

val canonical (#a:Type) (x:a) : prop

val canonical_f (a:Type) (#_ : (exists (x : a). True)) : a

val canonical_eq
  (#a:Type) (x : a)
  : Lemma (requires canonical x)
          (ensures x == canonical_f a)
          [SMTPat (canonical x)]
