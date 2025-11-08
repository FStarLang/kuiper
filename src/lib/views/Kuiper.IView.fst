module Kuiper.IView

#lang-pulse

let full_iff_cardinal_1
  (vw : aiview)
  (_ : Enumerable.enumerable vw.ait)
  : Lemma (requires is_full_view vw)
          (ensures Enumerable.cardinal vw.ait #_ == vw.len)
= let b : bijection vw.ait (natlt vw.len) = Kuiper.Bijection.bij_inj' vw.step.imap in
  Kuiper.Enumerable.bijection_implies_equal_cardinal _ _ b;
  ()

let full_iff_cardinal_2
  (vw : aiview)
  (_ : Enumerable.enumerable vw.ait)
  : Lemma (requires Enumerable.cardinal vw.ait #_ == vw.len)
          (ensures is_full_view vw)
= Kuiper.Enumerable.injection_equal_cardinal_implies_bijection _ _ vw.step.imap

let full_iff_cardinal
  (vw : aiview)
  {| d : Enumerable.enumerable vw.ait |}
  : Lemma (is_full_view vw <==> Enumerable.cardinal vw.ait #_ == vw.len)
= Classical.move_requires_2 (full_iff_cardinal_1) vw d;
  Classical.move_requires_2 (full_iff_cardinal_2) vw d

let it_nat_rel (vw : aiview) (i : vw.ait)
  (j : natlt vw.len{FStar.Functions.in_image vw.step.imap.f j})
  : Lemma (it_to_nat vw i == j <==> i == it_of_nat vw j)
  = ()

let full_view_bij (avw : aiview { is_full_view avw })
  : Ghost (avw.ait =~ natlt avw.len)
          (requires True)
          (ensures fun b -> forall x. b.ff x == it_to_nat avw x)
  = Kuiper.Bijection.bij_inj' avw.step.imap
