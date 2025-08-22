module Kuiper.Tag

#lang-pulse

open Pulse.Lib.Pervasives

let tagged (x:int) (s : slprop) : slprop = s

ghost
fn get_tag (x:int) (#s : slprop)
  preserves tagged x s
  returns  s' : slprop
  ensures  pure (s == s')
{
  s
}

ghost
fn tag (x:int) (s : slprop)
  requires s
  ensures  tagged x s
{
  fold tagged x s;
}

ghost
fn untag (x:int) (s : slprop)
  requires tagged x s
  ensures  s
{
  unfold tagged x s;
}
