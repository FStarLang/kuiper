module Kuiper.Kernel.Par
open Pulse
open Pulse.Lib.ConditionVar

#lang-pulse

fn par (#preL: slprop) #postL #preR #postR (vis: visibility) #l0 (l: loc_id)
  {| is_send_across vis preL, is_send_across vis postL |}
  (fn f () requires loc l ** preL ensures loc l ** postL)
  (fn g () requires preR ensures postR)
  preserves loc l0 ** pure (vis l0 == vis l)
  requires preL ** preR
  ensures postL ** postR
{
  on_intro preL;
  let c = create (on l0 postL) #_;
  fork' (on l0 preL ** send c (on l0 postL)) fn _ {
    impersonate unit l (on l0 preL) (fun _ -> on l0 postL) fn _ {
      is_send_across_elim vis preL #_ l;
      on_elim _;
      f ();
      on_intro postL;
      is_send_across_elim vis postL #_ l0;
    };
    signal c #(on l0 postL);
  };
  g ();
  wait c #(on l0 postL);
  on_elim postL;
}
