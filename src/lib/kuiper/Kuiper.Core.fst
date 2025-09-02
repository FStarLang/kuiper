module Kuiper.Core
#lang-pulse
open Kuiper
open Kuiper.Len

instance has_len_vec (a:Type) : has_len (Pulse.Lib.Vec.vec a) = {
  len = Pulse.Lib.Vec.length;
}

class has_core (p : slprop) = {
  _core_of : prop;
  _get_core : unit -> stt_ghost unit emp_inames p (fun _ -> p ** pure _core_of);
}

let core_of (p:slprop) {| d : has_core p |} : prop = d._core_of

ghost
fn get_core
  (p:slprop)
  {| d : has_core p |}
  requires p
  ensures  p ** pure (core_of p)
{
  let f = d._get_core;
  f ();
}

instance has_core_vec_pts_to
  (#t:Type) (a : Pulse.Lib.Vec.vec t) (#f:perm) (s : Seq.seq t)
: has_core (Pulse.Lib.Vec.pts_to a #f s)
=
{
  _core_of = (Pulse.Lib.Vec.length a == Seq.length s);
  _get_core = (fun () -> Pulse.Lib.Vec.pts_to_len a #f #s);
}

fn test (a : vec int{len a > 0})
  (#s : seq int)
  requires a |-> s
  returns  x : int
  ensures  a |-> s ** pure (Seq.length s > 0 /\ x == Seq.head s)
{
  open Pulse.Lib.Vec;
  (* need final underscore due to pulse bug *)
  get_core (a |-> s) #_;
  let v = a.(0sz);
  v
}
