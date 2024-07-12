module GPU.MatrixBarrier

open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open GPU.Base
module B = GPU.Barrier2

let mbarrier_tok
  (#n:nat)
  (p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  (b : B.barrier n)
  (it : nat)
  (tid : nat { 0 <= tid /\ tid < n })
  : slprop = B.barrier_tok #n
    (fun it (from : nat { 0 <= from /\ from < n }) -> bigstar 0 n (p it from))
    (fun it (to : nat { 0 <= to /\ to < n }) -> bigstar 0 n (fun (from : nat { 0 <= from /\ from < n }) -> p it from to)) b it tid

```pulse
ghost
fn mk_mbarrier_proof
  (n : nat)
  (p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  (it: nat)
  requires bigstar 0 n (fun (from : nat { 0 <= from /\ from < n }) -> bigstar 0 n (p it from))
  ensures  bigstar 0 n (fun (to : nat { 0 <= to /\ to < n }) -> bigstar 0 n (fun (from : nat { 0 <= from /\ from < n }) -> p it from to))
{
  bigstar_map #0 #0 #0 #n #(fun (from : nat { 0 <= from /\ from < n }) -> bigstar 0 n (p it from)) #_
    (fun (from : nat { 0 <= from /\ from < n }) -> bigstar_eta _);
  bigstar_commute #0 #0 0 n 0 n (fun (from : nat { 0 <= from /\ from < n }) -> fun (to : nat { 0 <= to /\ to < n }) -> p it from to);
}
```

// TODO: remove
```pulse
ghost fn fold_mbarrier_tok
  (#n:nat)
  (p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  (b : B.barrier n)
  (it : nat)
  (tid : nat { 0 <= tid /\ tid < n })
  requires B.barrier_tok #n
    (fun it (from : nat { 0 <= from /\ from < n }) -> bigstar 0 n (p it from))
    (fun it (to : nat { 0 <= to /\ to < n }) -> bigstar 0 n (fun (from : nat { 0 <= from /\ from < n }) -> p it from to)) b it tid
  ensures mbarrier_tok #n p b it tid
{
  fold (mbarrier_tok #n p b it tid)
}
```

```pulse
ghost
fn mk_mbarrier
  (n : nat)
  (p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  requires emp
  returns  b : erased (B.barrier n)
  ensures  bigstar 0 n (mbarrier_tok p b 0)
{
  let b = B.mk_barrier n
    (fun it (from : nat { 0 <= from /\ from < n }) -> bigstar 0 n (p it from))
    (fun it (to : nat { 0 <= to /\ to < n }) -> bigstar 0 n (fun (from : nat { 0 <= from /\ from < n }) -> p it from to))
    (mk_mbarrier_proof n p);
  bigstar_map #0 #0 #0 #n (fold_mbarrier_tok #n p b 0);
  b
}
```

// __syncthreads()
```pulse
fn mbarrier_wait
  (#n : erased nat)
  (#p : (it:nat -> from: nat { 0 <= from /\ from < n } -> to: nat { 0 <= to /\ to < n } -> slprop))
  (b : B.barrier n)
  (#it : erased nat)
  (#tid : erased nat { tid < n })
  requires mbarrier_tok p b  it    tid ** bigstar 0 n (p it tid)
  ensures  mbarrier_tok p b (it+1) tid ** bigstar 0 n (fun (from: nat { 0 <= from /\ from < n }) -> p it from tid)
{
  unfold mbarrier_tok;
  B.barrier_wait #n #(fun it (from: nat { 0 <= from /\ from < n }) -> bigstar 0 n (p it from)) #_ b;
  fold (mbarrier_tok p b (it+1) tid);
}
```

```pulse
ghost
val
fn drop_mbarrier
  (#n : nat)
  (#p : (it:nat -> from: nat { from < n } -> to: nat { to < n } -> slprop))
  (#b : B.barrier n)
  (#it: nat)
  requires bigstar 0 n (mbarrier_tok p b it)
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



