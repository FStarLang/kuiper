module Kuiper.Kernel.Softmax

(* Softmax, in two kernel calls.
   1. Compute sum of exps (does not modify array)
   2. Exp and divide by sum. Note: we are callng exp twice for every value.  An
   alernative would be to first do an exp pass, then reduce, then divide, but I
   think the extra kernel launch would be more expensive than the redundant
   exp's.
   *)

#lang-pulse

open Kuiper
module Vec = Pulse.Lib.Vec
module Array1 = Kuiper.Array1
open Kuiper.Array1

open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Softmax

let softmax_approx
    (#et:Type0) {| floating et, real_like et, floating_real_like et |}
    (s0:seq et) (r0:seq real { s0 %~ r0 /\ Seq.length r0 > 0 })
    (summ : et{summ %~ rsum (seq_map rexp r0)})
  : Lemma
    (ensures seq_map (fun x -> div x summ) (seq_map exp s0) %~ softmax_real r0)
  = let exps = seq_map rexp r0 in
    sum_non_zero exps 0.0R;
    ()

inline_for_extraction noextract
fn softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp{nth <= max_threads})
  (#lena : szp)
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va: erased (lseq et lena))
  (ra: erased (lseq real lena))
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
  ensures
    exists* (va' : lseq et lena).
      on gpu_loc (a |-> va') **
      pure (va' %~ softmax_real ra)
{
  (* Compute sum of exps (does not modify array) *)
  assert pure (Seq.equal (seq_map id (seq_map rexp ra)) (seq_map rexp ra));
  let sum = Kuiper.Kernel.HReduce.reduce exp rexp nth lena a ra;

  (* Exp and divide by sum. *)
  Kuiper.Kernel.Map.map_gpu (fun x -> div (exp x) sum) lena a;

  (* Note: this could also be fused into a single kcall. The comfortable way of
     doing that would require extensible barrier contracts, so we can add one more
     step to the barrier of HReduce. *)

  softmax_approx va ra sum;
  ()
}

inline_for_extraction noextract
fn softmax
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp{nth <= max_threads})
  (#lena : szp)
  (a : Vec.lvec et lena)
  (#va : erased (lseq et lena))
  (ra  : erased (lseq real lena))
  preserves
    cpu
  requires
    a |-> va **
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
  ensures
    exists* (va' : lseq et lena).
      a |-> va' **
      pure (va' %~ softmax_real ra)
{
  let ga = Array1.alloc0 #et lena (l1_forward lena);
  Array1.memcpy_host_to_device ga a lena;
  softmax_gpu nth ga ra;
  Array1.memcpy_device_to_host' a 0sz ga 0sz lena;
  Array1.free ga;
  ()
}
