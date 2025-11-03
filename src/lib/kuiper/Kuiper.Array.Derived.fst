module Kuiper.Array.Derived

#lang-pulse

open Pulse.Lib.Vec
open Pulse.Lib.WithPure
open Pulse
open FStar.Tactics.V2
open FStar.Seq
open Kuiper.Base
open Kuiper.Sized
open Kuiper.SizeT
open Kuiper.Seq.Common
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Divides { (/?+) }
module SZ = FStar.SizeT

open Kuiper.Array.Core

ghost
fn gpu_pts_to_ref
  (#a:Type u#0)
  (#sz:nat)
  (#f : perm)
  (x:gpu_array a sz)
  (#v : seq a)
  preserves x |-> Frac f v
  ensures  pure (Seq.length v == sz /\ SZ.fits sz)
{
  gpu_pts_to_slice_ref x 0 sz;
}

ghost
fn gpu_slice_split'
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (#s : erased (seq a))
  (i n : nat)
  (#_ : squash (0 <= n-i /\ n-i < length s))
  (m:nat)
  requires gpu_pts_to_slice arr #f i m s
  ensures  gpu_pts_to_slice arr #f i n (seq_take (n-i) s) ** gpu_pts_to_slice arr #f n m (seq_drop (n-i) s)
{
  assert pure (Seq.equal s (seq_take (n - i) s @+ seq_drop (n - i) s));
  rewrite gpu_pts_to_slice arr #f i m s
       as gpu_pts_to_slice arr #f i m (seq_take (n - i) s @+ seq_drop (n - i) s);
  gpu_slice_split arr #f i n m;
}

ghost
fn gpu_array_unslice_1'
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  requires forall+ (i: natlt sz). exists* v. gpu_pts_to_cell arr #f i v
  ensures  exists* v. pts_to arr #f v
{
  let ff = forevery_exists #(natlt sz) (fun i v -> gpu_pts_to_cell arr #f i v);
  let ss = Seq.init_ghost sz (fun i -> ff i);
  forevery_ext #(natlt sz) _ (fun i -> gpu_pts_to_cell arr #f i (ss @! i));
  gpu_array_unslice_1 arr;
}
