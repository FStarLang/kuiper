module Kuiper.PipelineCopy
#lang-pulse

open Pulse.Lib.Pledge
open Kuiper.Array.Vectorized
open FStar.Seq

open Kuiper
open Kuiper.Seq.Common { seq_blit }
module SZ = Kuiper.SizeT

(* An abstract tag identifying one linear chain of batches. Deliberately has no
   exposed structure and, in particular, no ordering: see the note on
   `get_batch` below for why that matters. *)
val lineage: Type0
type pipeline_batch_t = erased (lineage & nat)

val batch_live (b: pipeline_batch_t): slprop
val batch_committed (b: pipeline_batch_t): slprop
val batch_done (b: pipeline_batch_t): slprop

(* This below seems like nonsense to me because you can't actually 
  construct a program that has this misbehavior, but it is true that in some
  cases of the result of get_batch() you get unsound behavior.. so I'm not sure. *)
(* Mints a fresh batch chain. The intended model hands out a *fresh* lineage on
   every call; that freshness is a model-side obligation and is deliberately
   not exposed to clients, who learn nothing at all about `b`.

   The lineage tag is what makes unconstrained minting sound, and it is load
   bearing. With `pipeline_batch_t = erased nat` and a `batch_done_lower` that
   compares ids across mints, the interface has no model. Consider the CUDA
   program

     __pipeline_commit(); __pipeline_wait_prior(0);
     __pipeline_memcpy_async(dst, src, 16); __pipeline_commit();

   at whose end the copy is committed but never waited, so `pts_to_slice dst`
   must not hold. Under a flat-nat interface it admits two proofs, differing
   only in which ghost token is threaded through the same physical code:

     P1: get_batch() -> x; get_batch() -> y;
         commit #y; wait #y (* batch_done y *);
         copy tagged #x; commit #x;
         ghost if (x <= y) { batch_done_lower y x; redeem (* owns dst *) }

     P2: same prefix, roles of x and y swapped, branching on (y <= x).

   P1 is safe only if `x <= y` is false in every execution of the prefix
   `get_batch(); get_batch()`, and P2 only if `y <= x` is false in every
   execution of that same prefix. `<=` is total on nat, so no interpretation of
   batch_live/batch_committed/batch_done validates both: the model must commit
   to an ordering for a pair of ids before it can know which direction a later
   proof will need, and it can be wrong either way. (No single client derives
   False, but the interface having no model means nothing proved on top of it
   is backed by a soundness argument.)

   Tagging by lineage confines the ordering used by `batch_done_lower` to a
   single chain. Within a chain there is exactly one live token at any time --
   `pipeline_commit` consumes one and produces its successor -- so the nat
   ordering really is commit order, and since `pipeline_wait_all_prior` is a
   full flush, `batch_done (l,n)` does imply every `(l,m)`-tagged copy with
   m <= n has retired. Across chains the model only has to keep mints distinct,
   which fresh name generation achieves in every execution with no dependence
   on what the proof does later. Cross-lineage interference on the real
   hardware pipeline only ever retires copies earlier than the spec claims,
   which is a sound weakening.

   So: do not expose anything that lets a client relate two lineages -- no
   decidable equality, no ordering, no lemma concluding `l1 == l2`. A client
   can still classically case split on `l1 == l2`, which is harmless precisely
   because the model picks fresh uniformly.

   TODO: `pipeline_batch_t` still needs a thread identity. A
   `Kuiper.Barrier.contract` can redistribute arbitrary slprops across threads,
   so `batch_live` / `batch_committed` can be moved to a thread other than the
   one that issued the copies, while `__pipeline_commit` and
   `__pipeline_wait_prior` only ever act on the calling thread's pipeline.
   Concretely: thread 0 issues a copy and commits, hands `batch_committed b`
   over a `__syncthreads()`, and thread 1's wait drains its own (empty)
   pipeline and redeems the pledge with thread 0's DMA still in flight. This is
   exactly the hazard the CUDA docs warn about -- `__syncthreads()` does not
   complete other threads' async copies. `batch_done` should stay sendable,
   since "wait, then __syncthreads(), then everyone reads" is the idiom we want
   to keep expressible.

   TODO: consider taking the initial token from the kernel launch instead
   (a linear per-thread token in `kpre`/`kpost`, like
   `Kuiper.Barrier.barrier_tok` in `kernel_desc`). That subsumes both TODOs
   above and additionally forces a thread to quiesce its pipeline before
   exiting the kernel, rather than leaving copies in flight into shared memory
   that is about to be reclaimed.

   The flat-nat problem described above used to apply verbatim to `get_epoch` in
   Kuiper.Epoch. That is now fixed: `get_epoch` is gone, and `init_epoch` mints
   a stream's epoch counter exactly once by consuming the exclusive
   `stream_fresh` token produced by `fresh_stream`. *)
ghost
fn get_batch ()
  returns b : pipeline_batch_t
  ensures batch_live b


// LATER: enforce global <-> shared memory requirement
noextract (* prevents krml warning *)
fn array_vec_cpy_pipelined
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (dst_arr : array et)
  (dst_off : SZ.t)
  (#dst_slice_i : erased nat)
  (#dst_slice_j : erased nat)
  (src_arr : array et)
  (src_off : SZ.t)
  (#src_slice_i : erased nat)
  (#src_slice_j : erased nat)
  (#f : perm)
  (#ss : erased (seq et))
  (#ds : erased (seq et))
  (#_ : squash (dst_slice_i <= dst_off /\ dst_off + chunk et <= dst_slice_j))
  (#_ : squash (Seq.length ds == dst_slice_j - dst_slice_i))
  (#_ : squash (src_slice_i <= src_off /\ src_off + chunk et <= src_slice_j))
  (#_ : squash (Seq.length ss == src_slice_j - src_slice_i))
  (#b: pipeline_batch_t)
  preserves gpu
  requires pts_to_slice src_arr #f src_slice_i src_slice_j ss
  preserves batch_live b
  requires  pure (aligned' 16 src_arr src_off)
  requires  pure (aligned' 16 dst_arr dst_off)
  requires  pts_to_slice dst_arr dst_slice_i dst_slice_j ds
  ensures
    exists* s'.
      (pledge0 (batch_done b) (
        (pts_to_slice dst_arr dst_slice_i dst_slice_j s') **
        (pts_to_slice src_arr #f src_slice_i src_slice_j ss)
      )) **
      pure (s' == seq_blit ds (dst_off - dst_slice_i) ss (src_off - src_slice_i) (chunk et))

noextract 
fn pipeline_commit
  (#b: pipeline_batch_t)
  preserves gpu
  requires batch_live b
  returns b' : pipeline_batch_t
  ensures 
    batch_committed b **
    batch_live b' **
    pure (fst b' == fst b /\ snd b' > snd b)

noextract
fn pipeline_wait_all_prior
  (#b: pipeline_batch_t)
  preserves gpu
  requires batch_committed b
  ensures batch_done b

ghost
fn batch_done_lower (b1 b2 : pipeline_batch_t)
  preserves batch_done b1
  requires pure ((fst b2 == fst b1) /\ (snd b2 <= snd b1))
  ensures  batch_done b2