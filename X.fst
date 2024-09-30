module X

#lang-pulse
open Pulse

fn foo (r : ref int)
  requires pts_to r 2
  ensures  pts_to r 2
{
  with #w. assert pts_to r w;
}
