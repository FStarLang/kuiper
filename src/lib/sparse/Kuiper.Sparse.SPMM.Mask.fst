module Kuiper.Sparse.SPMM.Mask

#lang-pulse

open Kuiper
open Kuiper.Sparse
open Kuiper.Array.Vectorized
open Kuiper.Array2.Vectorized
open Kuiper.EMatrix

inline_for_extraction noextract
fn mask_cell
  (#et : Type0)
  (#n : sz)
  (x : larray et n)
  (i : szlt n)
  (z : et)
  preserves gpu
  requires array_live_cell x i
  ensures  pts_to_cell x i z
{
  unfold array_live_cell x;
  slice_write x i z;
  with s. assert pts_to_slice x i (i + 1) s;
  assert pure (Seq.equal s seq![z]);
}

// TODO podria generalizar y tomar from y to, aunque cuando
// repartimos por threads es medio raro
// Quizas cambiar la def de thread_pts_to

inline_for_extraction noextract
fn mask_array
  (#et : Type0)
  (#n : sz)
  (x : larray et n)
  (to : szle n)
  (z : et)
  (nthr : sz)
  (tid : szlt nthr)
  preserves gpu
  requires thread_slice_live x 0 to nthr tid
  requires pure (fits (n + nthr))
  ensures thread_slice_pts_to_value x 0 to z nthr tid
{
  unfold thread_slice_live x 0 to nthr tid;

  let to_ : sz = (to +^ (nthr -^ 1sz) -^ tid) /^ nthr;
  forevery_rw_size ((to - 0 - tid) `divup` nthr) to_;

  foreach to_
    (fun k -> array_live_cell x (0 + k * nthr + tid))
    (fun k -> Cell (x <: array et) (k * nthr + tid <: nat) |-> z)
    #gpu
    fn k {
      rewrite each (0 + k * nthr + tid) as (k * nthr + tid);
      mask_cell x (k *^ nthr +^ tid) z;
    };

  forevery_rw_size to_ ((to - 0 - tid) `divup` nthr);

  forevery_ext #(natlt ((to - 0 - tid) `divup` nthr))
    (fun k -> Cell (x <: array et) (k * nthr + tid <: nat) |-> z)
    (fun k -> Cell (x <: array et) (0 + k * nthr + tid <: nat) |-> z);

  fold thread_slice_pts_to_value x 0 to z nthr tid;
}