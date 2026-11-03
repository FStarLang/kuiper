module Kuiper.Example.LoopAlias

#lang-pulse

open Pulse.Lib.Pervasives
open Pulse.Lib.Reference
module U32 = FStar.UInt32

(* The two references name the same cell. Extraction must preserve the
   saved value across a write through the other reference. *)
fn alias_step (p q : ref U32.t)
  requires p |-> 0ul ** pure (p == q)
  ensures p |-> 1ul
{
  let mut again = true;
  while (!again)
    invariant exists* (v:U32.t) (b:bool).
      p |-> v ** again |-> b ** pure (v == (if b then 0ul else 1ul))
    decreases (if !again then 1 else 0)
  {
    let old = !p;
    rewrite (p |-> old) as (q |-> old);
    q := 5ul;
    rewrite (q |-> 5ul) as (p |-> 5ul);
    again := false;
    p := U32.add old 1ul;
  };
}
