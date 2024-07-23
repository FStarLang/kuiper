module GPU.Barrier2

open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open GPU.Base
open GPU.SizeT
module SZ = FStar.SizeT

[@@erasable]
val barrier
  (n:nat)
  : Type0

val barrier_tok
  (#n:nat)
  (p : (it:nat -> tid:nat { 0 <= tid /\ tid < n } -> slprop))
  (q : (it:nat -> tid:nat { 0 <= tid /\ tid < n } -> slprop))
  (b : barrier n)
  (it : nat)
  (tid : nat { tid < n })
  : slprop

```pulse
ghost
val
fn mk_barrier
  (n: SZ.t { 0 < n /\ n <= max_threads })
  (p : (it:nat -> tid:nat { 0 <= tid /\ tid < n } -> slprop))
  (q : (it:nat -> tid:nat { 0 <= tid /\ tid < n } -> slprop))
  (pf : (it:nat -> stt_ghost unit emp_inames
                  (requires bigstar 0 n (p it))
                  (ensures  fun _ -> bigstar 0 n (q it))))
  requires block_setup n
  returns  b : erased (barrier n)
  ensures  block_setup n ** bigstar 0 n (barrier_tok p q b 0)
```

// __syncthreads()
```pulse
val fn barrier_wait
  (#n : erased nat)
  (#p : (it:nat -> tid:nat { 0 <= tid /\ tid < n } -> slprop))
  (#q : (it:nat -> tid:nat { 0 <= tid /\ tid < n } -> slprop))
  (b : barrier n)
  (#it : erased nat)
  (#tid : erased nat { tid < n })
  requires barrier_tok p q b  it    tid ** p it tid
  ensures  barrier_tok p q b (it+1) tid ** q it tid
```

```pulse
ghost
val
fn drop_barrier
  (#n : nat)
  (#p : (it:nat -> tid:nat { 0 <= tid /\ tid < n } -> slprop))
  (#q : (it:nat -> tid:nat { 0 <= tid /\ tid < n } -> slprop))
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



