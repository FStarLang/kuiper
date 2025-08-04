module Kuiper.IView

#lang-pulse

let full_iff_cardinal
  (#len : nat)
  (vw : aiview len)
  : Lemma (is_full_view vw <==> vw.ait_enum._cardinal == len)
          [SMTPat (is_full_view vw)]
  = admit()