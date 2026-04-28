module Kuiper.Kernel.Softmax

(* Much of this module is layout-polymorphic, but
we fix to l1_forward at the end so we can use arr_read_1. This
should not be the case once there are more flexible memcpy's. *)

// This is a very naive implementation of softmax on the GPU,
// which uses three separate kernels launches (exp, reduce, divide).
// A fused version is possible.

#lang-pulse

open Kuiper
module Vec = Pulse.Lib.Vec
module Array1 = Kuiper.Array1
open Kuiper.Array1

open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }

let softmax_approx
    (#et:Type0) {| floating et, real_like et, floating_real_like et |}
    (s0:seq et) (r0:seq real { s0 %~ r0 /\ Seq.length r0 > 0 })
    (summ : et{summ %~ rsum (seq_map rexp r0)})
  : Lemma
    (ensures seq_map (fun x -> div x summ) (seq_map exp s0) %~ softmax_real r0)
  = let exps = seq_map rexp r0 in
    sum_non_zero exps 0.0R;
    Classical.forall_intro_2 (fun x -> Classical.move_requires (exp_approx #et x));
    Classical.forall_intro_4 (fun (x y : et) (r : real) -> Classical.move_requires (div_approx #et x y r));
    ()

inline_for_extraction noextract
fn softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp{nth <= max_threads})
  (#lena : szp)
  (a : array1 et (l1_forward lena) { is_global a })
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
  Classical.forall_intro_2 (fun x -> Classical.move_requires (exp_approx #et x));
  assert pure (Seq.equal (seq_map id (seq_map rexp ra)) (seq_map rexp ra));
  let sum = Kuiper.Kernel.HReduce.reduce exp rexp nth lena a ra;

  (* Exp and divide by sum. Note: we are callng exp twice for every value.  An
  alernative would be to first do an exp pass, then reduce, then divide, but I
  think the extra kernel launch would be more expensive than the redundant
  exp's. *)
  Kuiper.Kernel.Map.map_gpu (fun x -> div (exp x) sum) lena a;

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
