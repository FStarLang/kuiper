module Kuiper.IView

#lang-pulse

let full_iff_cardinal
  (vw : aiview)
  : Lemma (is_full_view vw <==> vw.sch.ait_enum._cardinal == vw.len)
          [SMTPat (is_full_view vw)]
  = admit()
