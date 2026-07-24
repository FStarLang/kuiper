module Kuiper.Kernel.Par
open Pulse
open Pulse.Lib.ConditionVar

#lang-pulse

divergent
fn impersonate_div
    u#a (a: Type u#a)
    (l: loc_id) (pre: slprop) (post: a -> slprop)
    {| placeless pre, ((x:a) -> placeless (post x)) |}
    (f: unit -> stt_div a (loc l ** pre) (fun x -> loc l ** post x))
  requires pre
  returns x: a
  ensures post x

divergent
fn par (#preL: slprop) #postL #preR #postR (vis: visibility) #l0 (l: loc_id)
  {| is_send_across vis preL, is_send_across vis postL |}
  (f : divergent fn () requires loc l ** preL ensures loc l ** postL)
  (g : divergent fn () requires preR ensures postR)
  preserves loc l0 ** pure (vis l0 == vis l)
  requires preL ** preR
  ensures postL ** postR
