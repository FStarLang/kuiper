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
module SZ = Kuiper.SizeT
open Kuiper.Real { log }
open Kuiper.Seq.Common
module KS = Kuiper.Spec.Softmax
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }

let log_softmax_approx
    (#et:Type0) {| floating et, real_like et, floating_real_like et |}
    (#n:nat) (va:chest1 et n) (ra:chest1 real n { va %~ ra /\ n > 0 })
    (summ : et { summ %~ chest1_rsum (chest_map exp ra) })
  : Lemma (ensures chest_map (fun x -> x `sub` flog summ) va %~ log_softmax_real ra)
  = let aux (i : natlt n)
      : Lemma (acc1 (chest_map (fun x -> x `sub` flog summ) va) i
               %~ acc1 (log_softmax_real ra) i)
      = () in
    Classical.forall_intro aux



inline_for_extraction noextract
fn log_softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp{nth <= max_threads})
  (#lena : szp)
  (a : array1 et (l1_forward lena) { is_global a })
  (#va: chest1 et lena)
  (ra: chest1 real lena)
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
  ensures
    exists* (va' : chest1 et lena).
      on gpu_loc (a |-> va') **
      pure (va' %~ log_softmax_real ra)
{
  assert pure (SZ.fits (lena + nth));
  (* Pointwise exp + compute sum (reduce preserves a). *)
  let sum = Kuiper.Kernel.Reduce.reduce1 fexp exp lena nth a ra;

  (* Compute pointwise log softmax. *)
  Kuiper.Kernel.Map.map_gpu (fun x -> x `sub` flog sum) lena a;

  log_softmax_approx va ra sum;
  ()
}

inline_for_extraction noextract
fn log_softmax
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
      pure (va' %~ chest1_to_seq (log_softmax_real (seq_to_chest1 ra)))
{
  let ga = alloc0 #et lena (l1_forward lena);
  with em. assert on gpu_loc (ga |-> em);
  map_loc gpu_loc #(ga |-> em) #(core ga |-> to_seq (l1_forward lena) em)
    fn _ { tensor_concr ga; };
  gpu_memcpy_host_to_device (core ga) a lena;
  map_loc gpu_loc #(core ga |-> reveal va) #(ga |-> from_seq (l1_forward lena) va)
    fn _ {
      tensor_abs' (l1_forward lena) (core ga);
      rewrite (from_array (l1_forward lena) (core ga) |-> from_seq (l1_forward lena) va)
           as (ga |-> from_seq (l1_forward lena) va);
    };
  log_softmax_gpu nth ga (seq_to_chest1 ra);
  with res. assert on gpu_loc (ga |-> res);
  map_loc gpu_loc #(ga |-> res) #(core ga |-> to_seq (l1_forward lena) res)
    fn _ { tensor_concr ga; };
  gpu_memcpy_device_to_host a (core ga) lena;
  map_loc gpu_loc #(core ga |-> to_seq (l1_forward lena) res) #(ga |-> res)
    fn _ {
      tensor_abs (l1_forward lena) (core ga);
      rewrite (from_array (l1_forward lena) (core ga) |-> reveal res)
           as (ga |-> reveal res);
    };
  free ga;
  ()
}


