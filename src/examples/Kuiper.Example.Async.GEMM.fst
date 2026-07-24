module Kuiper.Example.Async.GEMM

#lang-pulse

(* This computes (A*B)*(C*D) calling A*B and C*D asynchronously. *)

open Kuiper
open Pulse.Lib.Pledge
open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module N = Kuiper.Kernel.GEMM.Naive2

[@@allow_ambiguous]
ghost
fn redeem1 (s: stream_t) (e e' : epoch_t) (post : slprop)
  requires epoch_done s e' ** pledge0 (epoch_done s e) post ** pure (e' >= e)
  ensures  epoch_done s e' ** post
{
  done_lower s e' e;
  unfold pledge0;
  redeem_pledge _ _ _;
  drop_ (epoch_done s e);
}

let my_layout = l2_row_major 1024 1024

// Should not be needed.
inline_for_extraction noextract
instance c : ctlayout my_layout = c_l2_row_major 1024 1024sz

(* Fixing a size and a type, this is just for illustration *)
fn main (a b c d r : tensor f32 my_layout)
  (#eA #eB #eC #eD #eR : chest2 f32 1024 1024)
  preserves cpu
  preserves on gpu_loc <| a |-> eA ** b |-> eB ** c |-> eC ** d |-> eD
  requires  pure (is_global a /\ is_global b /\ is_global c /\ is_global d /\ is_global r)
  requires  on gpu_loc <| r |-> eR
  ensures   on gpu_loc <| r |-> MS.matmul (MS.matmul eA eB) (MS.matmul eC eD)
{
  let str1 = fresh_stream ();
  let str2 = fresh_stream ();
  let e1 = get_epoch str1 ();
  let e2 = get_epoch str2 ();

  (* Begin computing A*B -> s1 *)
  let s1 = alloc0 #f32 (1024sz *^ 1024sz) my_layout;
  launch (N.kdesc #f32 (fun _ n -> n) #1024sz #1024sz #1024sz a b s1) str1;

  (* Begin computing C*D -> s2 *)
  let s2 = alloc0 #f32 (1024sz *^ 1024sz) my_layout;
  launch (N.kdesc #f32 (fun _ n -> n) #1024sz #1024sz #1024sz c d s2) str2;

  (* Synchronize *)
  sync_stream str1;
  sync_stream str2;
  redeem1 str1 _ _ _;
  redeem1 str2 _ _ _;

  (* The expressions in the context are more complicated as they are built from
  mmcomb specs. We can rewrite since it's trivial to show that it's pointwise
  equal to the desired result. *)
  with es1'. assert on gpu_loc (s1 |-> es1');
  assert pure (Kuiper.EMatrix.equal es1' (MS.matmul eA eB));
  rewrite on gpu_loc (s1 |-> es1') as on gpu_loc (s1 |-> MS.matmul eA eB);
  with es2'. assert on gpu_loc (s2 |-> es2');
  assert pure (Kuiper.EMatrix.equal es2' (MS.matmul eC eD));
  rewrite on gpu_loc (s2 |-> es2') as on gpu_loc (s2 |-> MS.matmul eC eD);

  (* At this point, s1 and s2 point to the partial results. Multiply them (sync,
  as it's the last step) *)
  launch_sync (N.kdesc #f32 (fun _ n -> n) #1024sz #1024sz #1024sz s1 s2 r);

  (* We now have computed r. Same rewrite. *)
  with eR'. assert on gpu_loc (r |-> eR');
  assert pure (Kuiper.EMatrix.equal eR' (MS.matmul (MS.matmul eA eB) (MS.matmul eC eD)));
  rewrite on gpu_loc (r |-> eR') as on gpu_loc (r |-> MS.matmul (MS.matmul eA eB) (MS.matmul eC eD));

  (* Free swaps *)
  free s1;
  free s2;

  (* Drop ghost state *)
  drop_ (epoch_done str1 e1);
  drop_ (epoch_done str2 e2);
  with e. assert (epoch_live str1 e); drop_ (epoch_live str1 e);
  with e. assert (epoch_live str2 e); drop_ (epoch_live str2 e);

  (* Destroy streams *)
  destroy_stream str1;
  destroy_stream str2;
  ()
}
