module Kuiper.Injection

let __inj_cardinal (n1 n2 : nat)
  (i : natlt n1 @~> natlt n2)
  : Lemma (ensures n1 <= n2)
  = if n1 > n2 then
      Kuiper.Functions.pigeon n1 n2 i.f

let inj_cardinal (n1 n2 : nat)
  : Lemma (requires exists (b : natlt n1 @~> natlt n2). True)
          (ensures n1 <= n2)
  = Classical.forall_intro (__inj_cardinal n1 n2)
