module Kuiper.IView

#lang-pulse

let full_iff_cardinal_1
  (vw : aiview)
  : Lemma (requires is_full_view vw)
          (ensures vw.sch.ait_enum._cardinal == vw.len)
= let b : bijection vw.sch.ait (natlt vw.len) = Kuiper.Bijection.bij_inj' vw.step.imap in
  Kuiper.Enumerable.bijection_implies_equal_cardinal _ _ b;
  ()

let full_iff_cardinal_2
  (vw : aiview)
  : Lemma (requires vw.sch.ait_enum._cardinal == vw.len)
          (ensures is_full_view vw)
= Kuiper.Enumerable.injection_equal_cardinal_implies_bijection _ _ vw.step.imap

let full_iff_cardinal
  (vw : aiview)
  : Lemma (is_full_view vw <==> vw.sch.ait_enum._cardinal == vw.len)
= Classical.move_requires (full_iff_cardinal_1) vw;
  Classical.move_requires (full_iff_cardinal_2) vw

let full_view_bij (avw : aiview { is_full_view avw })
  : Ghost (avw.sch.ait =~ natlt avw.len)
          (requires True)
          (ensures fun b -> forall x. b.ff x == it_to_nat avw x)
  = Kuiper.Bijection.bij_inj' avw.step.imap
