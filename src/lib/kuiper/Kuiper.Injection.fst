module Kuiper.Injection

let lem_pat (#a #b : _) (d : a @~> b) (x y : a)
  : Lemma (d.f x == d.f y ==> x == y)
  = d.is_inj x y

let lem_forall_pat (#a #b : _) (d : a @~> b)
  : Lemma (forall x y. d.f x == d.f y ==> x == y)
  = ()

let __inj_cardinal (n1 n2 : nat)
  (i : natlt n1 @~> natlt n2)
  : Lemma (ensures n1 <= n2)
  = if n1 > n2 then
      Kuiper.Functions.pigeon n1 n2 i.f

let inj_cardinal (n1 n2 : nat)
  : Lemma (requires exists (b : natlt n1 @~> natlt n2). True)
          (ensures n1 <= n2)
  = Classical.forall_intro (__inj_cardinal n1 n2)
