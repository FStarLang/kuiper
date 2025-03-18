module WithPureTest
#lang-pulse

open Pulse.Nolib

ghost
fn intro_with_pure
  (p : prop)
  (v : squash p -> slprop)
  (_ : squash p)
  requires pure p ** v ()
  ensures  emp
{
  assert (v ());
  assert (exists* s. v s);
  with s. assert v s;
  drop_ (v s);
}
