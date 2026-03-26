module Kuiper.Async.GEMM

#lang-pulse

(* This computes (A*B)*(C*D) calling A*B and C*D asynchronously. *)

open Kuiper
open Pulse.Lib.Pledge
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs
module MS = Kuiper.Spec.GEMM
module N = Kuiper.Poly.GEMM.Naive

[@@allow_ambiguous]
ghost
fn redeem1 (e e' : erased nat) (post : slprop)
  requires epoch_done e' ** pledge0 (epoch_done e) post ** pure (e' >= e)
  ensures  epoch_done e' ** post
{
  done_lower e' e;
  unfold pledge0;
  redeem_pledge _ _ _;
  drop_ (epoch_done e);
}

inline_for_extraction noextract
instance c : clayout (row_major 1024 1024) =
  crepr_row_major.map 1024sz 1024sz

(* Fixing a size and a type, this is just for illustration *)
fn main (a b c d r : gpu_matrix f32 (row_major 1024 1024))
  (#eA #eB #eC #eD #eR : ematrix f32 1024 1024)
  preserves cpu
  preserves on gpu_loc (a |-> eA ** b |-> eB ** c |-> eC ** d |-> eD)
  requires  pure (is_global_matrix a /\ is_global_matrix b /\ is_global_matrix c /\ is_global_matrix d /\ is_global_matrix r)
  requires  on gpu_loc (r |-> eR)
  ensures   on gpu_loc (r |-> MS.matmul (MS.matmul eA eB) (MS.matmul eC eD))
{
  let e1 = get_epoch ();

  (* Begin computing A*B -> s1 *)
  let s1 = gpu_matrix_alloc0 #f32 1024sz 1024sz (row_major 1024 1024);
  launch (N.kdesc #f32 (fun _ n -> n) #1024sz #1024sz #1024sz a b s1);

  (* Begin computing C*D -> s2 *)
  let s2 = gpu_matrix_alloc0 #f32 1024sz 1024sz (row_major 1024 1024);
  launch (N.kdesc #f32 (fun _ n -> n) #1024sz #1024sz #1024sz c d s2);

  (* Synchronize *)
  sync_device ();
  redeem1 _ _ _;
  redeem1 _ _ _;

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
  gpu_matrix_free s1;
  gpu_matrix_free s2;

  (* Drop ghost state *)
  drop_ (epoch_done e1);
  with e. assert (epoch_live e); drop_ (epoch_live e);

  ()
}
