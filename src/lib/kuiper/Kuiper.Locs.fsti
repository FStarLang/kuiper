module Kuiper.Locs

#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open Pulse.Lib.Send
module T = FStar.Tactics.V2
open Pulse.Lib.Array.Core { visibility }

val gpu_of : visibility
val gpu_of_idem (l:loc_id) : Lemma (gpu_of (gpu_of l) == l)
val gpu_id_of : loc_id -> GTot int

val block_of : visibility
val block_of_idem (l:loc_id) : Lemma (block_of (block_of l) == l)
val block_id_of : loc_id -> GTot int

val gpu_id_loc (gpu_id:int) : l:loc_id { gpu_of l == l }
val gpu_id_loc_lemma (gpu_id:int) : Lemma
  (let l = gpu_id_loc gpu_id in
    gpu_id_of l == gpu_id
  )
let gpu_loc = gpu_id_loc 0

val block_id_loc (#[T.exact (`0)]gpu_id:int) (bid:int)
: l:loc_id { gpu_of l == gpu_id_loc gpu_id }
val block_id_loc_lemma (#[T.exact (`0)]gpu_id:int) (bid:int) : Lemma
  (let l = block_id_loc #gpu_id bid in
    block_id_of l == bid /\ block_of l == l
  )

val thread_id_loc (#[T.exact (`0)]gpu_id:int) (bid tid:int)
: l:loc_id { block_of l == block_id_loc #gpu_id bid /\ gpu_of l == gpu_id_loc gpu_id }
val thread_id_of (l:loc_id) : GTot int
val thread_id_loc_lemma (#[T.exact (`0)]gpu_id:int) (bid tid:int) : Lemma
  (let l = thread_id_loc #gpu_id bid tid in
    thread_id_of l == tid /\ block_id_of l == bid /\ gpu_id_of l == gpu_id
  )

//locations that agree on their blocks are on the same gpu
val block_of_same_gpu (l0 l1:_{block_of l0 == block_of l1})
: Lemma (gpu_of l0 == gpu_of l1)

(* Refinement of equivalence relations: g' is at least as fine as g. If a
   resource is sendable across g (can move within each g-class), it is sendable
   across any finer g' (smaller classes), since every g'-equal pair is g-equal.

   It is kept ABSTRACT (no unfolding to a quantifier) so that the SMT patterns
   below stay narrow: they fire only on [vis_refines _ _] terms, which appear
   only in sendability refinements, and never pollute unrelated (e.g. tiling
   arithmetic) proof contexts. *)
val vis_refines (#b:Type0) (g' g : loc_id -> b) : prop

(* Defining property, used by [weaken]. *)
val vis_refines_elim (#b:Type0) (g' g : loc_id -> b) (l l':loc_id)
: Lemma (requires vis_refines g' g /\ g' l == g' l') (ensures g l == g l')

(* Every relation refines itself; covers the "home visibility" and the global
   array at gpu_of cases. *)
val vis_refines_refl (g : visibility)
: Lemma (vis_refines g g) [SMTPat (vis_refines g g)]

(* block_of refines gpu_of (same block => same gpu): covers a global array sent
   across blocks. The variable trigger [vis_refines block_of g] stays narrow
   (only fires on vis_refines terms) and the guard g == gpu_of is discharged
   from is_global. *)
val vis_refines_block_gpu (g : visibility)
: Lemma (requires g == gpu_of) (ensures vis_refines block_of g)
        [SMTPat (vis_refines block_of g)]

(* The general weakening principle. This subsumes the old gpu_of -> block_of
   lift (which was just [weaken] specialized to g=gpu_of, g'=block_of), and is
   the single mechanism used to derive every concrete sendability from a
   resource's home visibility. It is NOT an instance (an unconditional
   gpu_of->block_of instance would be picked ahead of the per-resource
   weakening instances for concrete block arrays and then fail). It is invoked
   explicitly: by the per-resource weakening instances, and by [send_gpu_block]
   below for abstract slprops. *)
let weaken (#b:Type0) (#g #g':loc_id -> b) (#p:slprop)
      (h:is_send_across g p)
      (pf:squash (vis_refines g' g))
: is_send_across g' p
= fun l0 l1 -> vis_refines_elim g' g l0 l1; h l0 l1

(* Explicit gpu_of -> block_of weakening for *abstract* slprops, where no
   per-resource instance applies (e.g. the polymorphic kpre/kpost in the kernel
   cast machinery). Not an instance, for the reason above; call it by hand. *)
let send_gpu_block (p:slprop) (sp:is_send_across gpu_of p)
: is_send_across block_of p
= weaken sp ()

instance cond_sendable (b:bool) (p q:slprop)
      (vis:loc_id -> 'a)
      (f:is_send_across vis p)
      (g:is_send_across vis q)
: is_send_across vis (Pulse.Lib.Primitives.cond b p q)
= fun l0 l1 -> if b then f l0 l1 else g l0 l1

(* Token for being in GPU code *)
[@@no_mkeys]
let gpu (#[T.exact (`0)] gpu_id:int) : slprop =
  exists* (l:loc_id). loc l ** pure (gpu_of l == gpu_id_loc gpu_id /\ gpu_id_of l == gpu_id)

(* Token given to a particular block within a grid. Both here
and in thread_id, the first argument is always positive
when this resource is actually live, but not placing that refinement
here helps with inference in some places. *)
[@@no_mkeys]
let block_id (nblk : int) (bid : int) : slprop =
  exists* (l:loc_id). loc l ** pure (block_of l == block_id_loc bid /\ block_id_of l == bid)

(* Token given to a particular thread within a block *)
[@@no_mkeys]
let thread_id (nthr : int) (tid : int) : slprop =
  exists* (l:loc_id). loc l ** pure (thread_id_of l == tid)

val is_cpu_loc (l:loc_id) : prop

val is_cpu_loc_single_process (l0 l1:loc_id)
: Lemma (is_cpu_loc l0 /\ is_cpu_loc l1 ==> process_of l0 == process_of l1)

(* Token for being in CPU code *)
let cpu : slprop = exists* l. loc l ** pure (is_cpu_loc l)

ghost
fn map_loc (loc:loc_id) (#p #q:slprop) (f : ghost fn () requires p ensures q)
  requires on loc p
  ensures  on loc q
{
  ghost_impersonate loc (on loc p) (on loc q) fn () {
    on_elim p;
    f();
    on_intro q;
  }
}

ghost
fn reduce_with_steps (p:slprop) (steps:_)
requires p
ensures norm steps p
{
  norm_spec steps p;
  rewrite p as (norm steps p);
}
