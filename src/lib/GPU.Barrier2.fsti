module GPU.Barrier2

open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open GPU.Base

[@@erasable]
val barrier
  (n:nat)
  : Type0

// val barrier_alive
//   (n:nat)
//   (p : (it:nat -> tid:nat -> slprop))
//   (q : (it:nat -> tid:nat -> slprop))
//   (it : nat)
//   (b : barrier n p q)
//   : slprop

val barrier_tok
  (#n:nat)
  (p : (it:nat -> tid:nat -> slprop))
  (q : (it:nat -> tid:nat -> slprop))
  (b : barrier n)
  (it : nat)
  (tid : nat)
  : slprop

```pulse
ghost
val
fn mk_barrier
  (n : nat)
  (p : (it:nat -> tid:nat -> slprop))
  (q : (it:nat -> tid:nat -> slprop))
  (pf : (it:nat -> stt_ghost unit emp_inames
                  (requires bigstar 0 n (p it))
                  (ensures  fun _ -> bigstar 0 n (q it))))
  requires emp
  returns  b : erased (barrier n)
  ensures  bigstar 0 n (barrier_tok p q b 0)
```

// __syncthreads()
```pulse
val fn barrier_wait
  (#n : erased nat)
  (#p : (it:nat -> tid:nat -> slprop))
  (#q : (it:nat -> tid:nat -> slprop))
  (b : barrier n)
  (#it : erased nat)
  (#i : erased nat)
  requires barrier_tok p q b  it    i ** p it i
  ensures  barrier_tok p q b (it+1) i ** q it i
```

```pulse
ghost
val
fn drop_barrier
  (#n : nat)
  (#p : (it:nat -> tid:nat -> slprop))
  (#q : (it:nat -> tid:nat -> slprop))
  (#b : barrier n)
  (#it: nat)
  requires bigstar 0 n (barrier_tok p q b it)
  ensures  emp
```

(* Does this always deadlock? *)
// if (tid % 2) {
//   ...
//   __syncthreads();
// } else {
//   ...
//   __syncthreads();
// }



