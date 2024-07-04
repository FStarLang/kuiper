module GPU.Barrier2

open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open GPU.Base

[@@erasable]
val barrier
  (n:nat)
  (p : (it:nat -> tid:nat -> slprop))
  (q : (it:nat -> tid:nat -> slprop))
  : Type0

val barrier_alive
  (n:nat)
  (p : (it:nat -> tid:nat -> slprop))
  (q : (it:nat -> tid:nat -> slprop))
  (it : nat)
  (b : barrier n p q)
  : slprop

val barrier_tok
  (#n:nat)
  (#p : (it:nat -> tid:nat -> slprop))
  (#q : (it:nat -> tid:nat -> slprop))
  (b : barrier n p q)
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
  returns  b : barrier n p q
  ensures  barrier_alive n p q 0 b ** bigstar 0 n (barrier_tok b)
```

// __syncthreads()
```pulse
val fn barrier_wait
  (#n : nat)
  (#p : (it:nat -> tid:nat -> slprop))
  (#q : (it:nat -> tid:nat -> slprop))
  (b : barrier n p q)
  (#it : erased nat)
  (#i : erased nat)
  requires barrier_alive n p q  it    b ** barrier_tok b i ** p it i
  ensures  barrier_alive n p q (it+1) b ** barrier_tok b i ** q it i
```

(* Does this always deadlock? *)
// if (tid % 2) {
//   ...
//   __syncthreads();
// } else {
//   ...
//   __syncthreads();
// }



