module GPU.Barrier

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open GPU.Base
open Pulse.Lib.BigStar

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

(*
fn mk_barrier
  (n : nat)
  (p : nat -> vprop)
  (q : nat -> vprop)
  (pf : unit -> ghost unit (requires bigstar 0 n p) (ensures bigstar 0 n q))
  requires emp
  returns  b : barrier n p q
  ensures  barrier_alive n p q b ** bigstar 0 n (barrier_tok b)
*)

// __syncthreads()
(*
fn barrier_wait
  (#n : nat)
  (#p : nat -> vprop)
  (#q : nat -> vprop)
  (b : barrier n p q)
  (#i : erased nat)
  requires barrier_alive n p q b ** barrier_tok b i ** p i
  ensures  barrier_alive n p q b ** barrier_tok b i ** q i
*)

(* Does this always deadlock? *)
// if (tid % 2) {
//   ...
//   __syncthreads();
// } else {
//   ...
//   __syncthreads();
// }



