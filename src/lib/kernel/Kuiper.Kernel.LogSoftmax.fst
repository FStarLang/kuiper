module Kuiper.Kernel.LogSoftmax

(*
This implements log_softmax, essentially softmax + a pointwise log,
but due to error concerns the result is computed differently.

Recall

    softmax(s) = exp(s) / sum(exp(s))

i.e. pointwise exponentiation + normalize. So

    log_softmax(s) = log(exp(s) / sum(exp(s)))

which can be rewritten to:

     log_softmax(s) = s - log(sum(exp(s)))

Shifting the original sequence by any scalar also does not change the
result (this is true of softmax, and therefore of log_softmax) so another
possible implementation option is to first subtract the maximum element
from every cell. We don't do that yet. *)

#lang-pulse

open Kuiper
module Vec = Pulse.Lib.Vec
module Array1 = Kuiper.Array1
open Kuiper.Array1

open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }

(* Alternative definition closer to what we compute. *)
let log_softmax_real' (s:Seq.seq real { Seq.length s > 0 }) =
  let exps = seq_map rexp s in
  let summ : real = rsum exps in
  lseq_map #_ #_ #(Seq.length s) (fun x -> x -. rlog summ) s

(* Should add enough exp/log facts to prove this. *)
let real_log_softmax_lemma (s : seq real{len s > 0})
  : Lemma (log_softmax_real s == log_softmax_real' s)
  = let aux (i : natlt (len s)) : Lemma ((log_softmax_real s @! i) == (log_softmax_real' s @! i)) =
      calc (==) {
        log_softmax_real s @! i;
        == {}
        lseq_map rlog (seq_refine (fun x -> x >. 0.0R) (SM.softmax_real s))
          @! i;
        == {}
        rlog (seq_refine (fun x -> x >. 0.0R) (SM.softmax_real s) @! i);
        == { lem_seq_refine_at #real (fun x -> x >. 0.0R) (SM.softmax_real s) i } // FIXME: should be automatic
        rlog (SM.softmax_real s @! i);
        == {}
        rlog (rexp (s @! i) /. (rsum (seq_map rexp s)));
        == {}
        rlog (rexp (s @! i)) -. rlog (rsum (seq_map rexp s));
        == {}
        (s @! i) -. rlog (rsum (seq_map rexp s));
        == {}
        log_softmax_real' s @! i;
      };
      ()
    in
    Classical.forall_intro aux;
    assert (Seq.equal (log_softmax_real s) (log_softmax_real' s));
    ()

let log_softmax_approx
    (#et:Type0) {| floating et, real_like et, floating_real_like et |}
    (s0:seq et) (r0:seq real { s0 %~ r0 /\ Seq.length r0 > 0 })
    (summ : et{summ %~ rsum (seq_map rexp r0)})
  : Lemma (ensures seq_map (fun x -> x `sub` log summ) s0 %~ log_softmax_real r0)
  = let lhs = seq_map (fun x -> x `sub` log summ) s0 in
    let aux (i : natlt (len r0)) : Lemma ((lhs @! i) %~ (log_softmax_real' r0 @! i)) =
      let x = s0 @! i in
      assert (x %~ (r0 @! i));
      assert ((summ <: et) %~ rsum (seq_map rexp r0));
      log_approx summ (rsum (seq_map rexp r0));
      assert (log summ %~ rlog (rsum (seq_map rexp r0)));
      sub_approx x (log summ) (r0 @! i) (rlog (rsum (seq_map rexp r0)));
      assert ((x `sub` log summ)   %~  ((r0 @! i) -. rlog (rsum (seq_map rexp r0))));
      ()
    in
    Classical.forall_intro aux;
    real_log_softmax_lemma r0;
    ()

inline_for_extraction noextract
fn softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lena : szp)
  (a : array1 et (l1_forward lena) { is_global a })
  (#va: erased (lseq et lena))
  (ra: erased (lseq real lena))
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (lena <= max_threads) **
    pure (va %~ ra)
  ensures
    exists* (va' : lseq et lena).
      on gpu_loc (a |-> va') **
      pure (va' %~ log_softmax_real ra)
{
  (* Copy original array. *)
  let a' = Array1.alloc0 #et lena (l1_forward lena);
  Array1.memcpy_device_to_device a' a lena;

  (* Pointwise exponentiation. *)
  Kuiper.Kernel.Map.map_gpu exp lena a';

  Classical.forall_intro_2 (fun x -> Classical.move_requires (exp_approx #et x));

  (* Compute sum *)
  let sum = Kuiper.Kernel.HReduce.reduce lena a' (seq_map rexp ra);
  Array1.free a';

  (* Compute pointwise log softmax. *)
  Kuiper.Kernel.Map.map_gpu (fun x -> x `sub` log sum) lena a;

  log_softmax_approx va ra sum;
  ()
}

inline_for_extraction noextract
fn log_softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (#lena : szp)
  (a : Vec.lvec et lena)
  (#va : erased (lseq et lena))
  (ra  : erased (lseq real lena))
  preserves
    cpu
  requires
    a |-> va **
    pure (lena <= max_threads) **
    pure (va %~ ra)
  ensures
    exists* (va' : lseq et lena).
      a |-> va' **
      pure (va' %~ log_softmax_real ra)
{
  let ga = Array1.alloc0 #et lena (l1_forward lena);
  Array1.memcpy_host_to_device ga a lena;
  softmax_gpu ga ra;
  Array1.memcpy_device_to_host' a 0sz ga 0sz lena;
  Array1.free ga;
  ()
}
