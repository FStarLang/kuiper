module Kuiper.IView

#lang-pulse

let full_iff_cardinal_1
  (vw : aiview)
  (_ : Enumerable.enumerable vw.sch.ait)
  : Lemma (requires is_full_view vw)
          (ensures Enumerable.cardinal vw.sch.ait #_ == vw.len)
= let b : bijection vw.sch.ait (natlt vw.len) = Kuiper.Bijection.bij_inj' vw.step.imap in
  Kuiper.Enumerable.bijection_implies_equal_cardinal _ _ b;
  ()

let full_iff_cardinal_2
  (vw : aiview)
  (_ : Enumerable.enumerable vw.sch.ait)
  : Lemma (requires Enumerable.cardinal vw.sch.ait #_ == vw.len)
          (ensures is_full_view vw)
= Kuiper.Enumerable.injection_equal_cardinal_implies_bijection _ _ vw.step.imap

let full_iff_cardinal
  (vw : aiview)
  {| d : Enumerable.enumerable vw.sch.ait |}
  : Lemma (is_full_view vw <==> Enumerable.cardinal vw.sch.ait #_ == vw.len)
= Classical.move_requires_2 (full_iff_cardinal_1) vw d;
  Classical.move_requires_2 (full_iff_cardinal_2) vw d

let it_nat_rel (vw : aiview) (i : vw.sch.ait)
  (j : natlt vw.len{FStar.Functions.in_image vw.step.imap.f j})
  : Lemma (it_to_nat vw i == j <==> i == it_of_nat vw j)
  = ()

let full_view_bij (avw : aiview { is_full_view avw })
  : Ghost (avw.sch.ait =~ natlt avw.len)
          (requires True)
          (ensures fun b -> forall x. b.ff x == it_to_nat avw x)
  = Kuiper.Bijection.bij_inj' avw.step.imap
