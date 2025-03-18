module Kuiper.InnerGhostLem
#lang-pulse

open Pulse

#push-options "--fuel 1 --ifuel 1"

assume val p : int -> slprop
assume val q : int -> slprop

fn setup ()
  requires p 1
  ensures  q 1
{
  ghost
  fn aux (i:int)
    requires p i
    ensures  q i
  {
    admit()
  };
  aux 1
}

#pop-options
