module Idea.BigStar.Extensionality

open Pulse.Lib.Pervasives
open GPU
open Pulse.Lib.BigStar

assume val f : int -> vprop

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
  (f : (i: nat{m <= i /\ i < n} -> vprop))
  (g : (i: nat{m <= i /\ i < n} -> vprop))
  (h : ((i: nat{m <= i /\ i < n}) -> squash (f i == g i)))
  : vprop_equiv (bigstar m n f) (bigstar m n g)
  = bigstar_congr m n m n f g (fun i -> h (i+m));
    vprop_equiv_refl (bigstar m n f)

// And have Pulse use it automatically?

(* Maybe as: *)
let equate_via (_:'a) : unit = ()
[@@equate_via bigstar_equiv]
val bigstar' 
  (m : nat)
  (n : nat {m <= n})
  (f : (i:nat { m <= i /\ i < n } -> vprop))
  : vprop
let bigstar' = bigstar

(* Or: *)
let equate_arg_via (_:'a) : unit = ()
val bigstar''
  ([@@@ equate_strict] m : nat)
  ([@@@ equate_strict] n : nat {m <= n})
  ([@@@ equate_arg_via bigstar_equiv] f : (i:nat { m <= i /\ i < n } -> vprop))
  : vprop
let bigstar'' = bigstar
