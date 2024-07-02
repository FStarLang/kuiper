module GPU.Barrier

open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open GPU.Base

val barrier
  (n:nat)
  (p : nat -> vprop)
  (q : nat -> vprop)
  : Type0

val barrier_alive
  (n:nat)
  (p : nat -> vprop)
  (q : nat -> vprop)
  (b : barrier n p q)
  : vprop

val barrier_tok
  (#n:nat)
  (#p : nat -> vprop)
  (#q : nat -> vprop)
  (b : barrier n p q)
  (tid : nat)
  : vprop

```pulse
val
fn mk_barrier
  (n : nat)
  (p : nat -> vprop)
  (q : nat -> vprop)
  (pf : unit -> stt_ghost unit emp_inames
                  (requires bigstar 0 n p)
                  (ensures  fun _ -> bigstar 0 n q))
  requires emp
  returns  b : barrier n p q
  ensures  barrier_alive n p q b ** bigstar 0 n (barrier_tok b)
```

// __syncthreads()
```pulse
ghost
val fn barrier_wait
  (#n : nat)
  (#p : nat -> vprop)
  (#q : nat -> vprop)
  (b : barrier n p q)
  (#i : erased nat)
  requires barrier_alive n p q b ** barrier_tok b i ** p i
  ensures  barrier_alive n p q b ** barrier_tok b i ** q i
```

(* Does this always deadlock? *)
// if (tid % 2) {
//   ...
//   __syncthreads();
// } else {
//   ...
//   __syncthreads();
// }



