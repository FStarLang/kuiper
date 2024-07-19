module Idea.BigStar.Extensionality

open Pulse.Lib.Pervasives
open GPU
open Pulse.Lib.BigStar

assume val f : int -> slprop

```pulse
fn test ()
  requires bigstar 0 9 f
  ensures  bigstar 0 9 (fun i -> f i)
{
  bigstar_extensionality 0 9 f (fun i -> f i) (fun _ -> ()); // boring
  ();
}
```

// Could we say:

let bigstar_equiv
  (m : nat)
  (n : nat {m <= n})
  (f : (i: nat{m <= i /\ i < n} -> slprop))
  (g : (i: nat{m <= i /\ i < n} -> slprop))
  (h : ((i: nat{m <= i /\ i < n}) -> squash (f i == g i)))
  : slprop_equiv (bigstar m n f) (bigstar m n g)
  = bigstar_congr #0 #0 m n m n f g (fun i -> h (i+m));
    assert (bigstar m n f == bigstar m n g);
    coerce_eq () <| slprop_equiv_refl (bigstar m n f)

// And have Pulse use it automatically?

(* Maybe as: *)
let equate_via (_:'a) : unit = ()
[@@equate_via bigstar_equiv]
val bigstar' 
  (m : nat)
  (n : nat {m <= n})
  (f : (i:nat { m <= i /\ i < n } -> slprop))
  : slprop
let bigstar' = bigstar

(* Or: *)
let equate_arg_via (_:'a) : unit = ()
val bigstar''
  ([@@@ equate_strict] m : nat)
  ([@@@ equate_strict] n : nat {m <= n})
  ([@@@ equate_arg_via bigstar_equiv] f : (i:nat { m <= i /\ i < n } -> slprop))
  : slprop
let bigstar'' = bigstar
