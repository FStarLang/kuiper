module Kuiper.Kernel.TMap

(* Simple kernel: pointwise map of a function on an array. *)

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
open Kuiper.Chest
open Kuiper.Shape
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }

ghost
fn setup
  (#et : Type0) (#r : nat) (#d : shape r)
  (n : sz{SZ.v n == sizeof d /\ n <= max_blocks * max_threads}) // sigh
  (f : et -> et)
  (#l : tlayout d)
  (a : tensor et l)
  (#s : chest d et)
  ()
  norewrite
  requires
    (a |-> s)
  ensures
    (forall+ (i : natlt n).
      Cell a (unflatten d i) |-> acc s (unflatten d i)) **
    pure (SZ.fits (tlayout_ulen l))
{
  tensor_pts_to_ref a;
  tensor_explode a;
  forevery_iso (flatten_bij d) (fun (i : abs d) -> Cell a i |-> acc s i);
  forevery_rw_size (sizeof d) (SZ.v n);
}

ghost
fn teardown
  (#et : Type0) (#r : nat) (#d : shape r)
  (n : sz{SZ.v n == sizeof d /\ n <= max_blocks * max_threads}) // sigh
  (f : et -> et)
  (#l : tlayout d)
  (a : tensor et l)
  (#s : chest d et)
  ()
  norewrite
  requires
    (forall+ (i : natlt n).
      Cell a (unflatten d i )|-> (f (acc s (unflatten d i)))) **
    pure (SZ.fits (tlayout_ulen l))
  ensures
    a |-> (Kuiper.Chest.chest_map f s)
{
  forevery_rw_size (SZ.v n) (sizeof d);
  forevery_iso_back (flatten_bij d) (fun (i : abs d) -> Cell a i |-> (f (acc s i)));
  forevery_map
    (fun (i : abs d) -> Cell a i |-> (f (acc s i)))
    (fun (i : abs d) -> Cell a i |-> ((chest_map f s) `acc` i))
    fn x { () };
  tensor_implode a;
  ()
}

inline_for_extraction noextract
fn kf
  (#et : Type0) (#r : erased nat) (#d : shape r) (cd : cshape d)
  (f : et -> et)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (#s : chest d et)
  (id : szlt (sizeof d))
  ()
  requires
    gpu **
    Cell a (unflatten d id) |-> (acc s (unflatten d id))
  ensures
    gpu **
    Cell a (unflatten d id) |-> (f (acc s (unflatten d id)))
{
  rewrite
    Cell a (unflatten d id) |-> (acc s (unflatten d id))
  as
    Cell a (up (cunflatten cd id)) |-> (acc s (unflatten d id));
  let x = tensor_read_cell a (cunflatten cd id);
  tensor_write_cell a (cunflatten cd id) (f x);
  rewrite
    Cell a (up (cunflatten cd id)) |-> f (acc s (unflatten d id))
  as
    Cell a (unflatten d id) |-> f (acc s (unflatten d id));
}

inline_for_extraction noextract
let kmap
  (#et : Type0) (#r : erased nat) (#d : shape r) (cd : cshape d)
  (f : et -> et)
  (#l : tlayout d) {| ctlayout l |}
  (n : sz{SZ.v n == sizeof d /\ n <= max_blocks * max_threads})
  (a : tensor et l)
  (#s : chest d et)
  (#_ : is_global a)
  : kernel_desc
      (requires a |-> s)
      (ensures  a |-> chest_map f s)
= {
    nthr = n;
    f = kf cd f a;

    frame    = pure (SZ.fits (tlayout_ulen l));
    setup    = setup    n f a #s;
    teardown = teardown n f a #s;
    kpre  = (fun (i : natlt (sizeof d)) -> Cell a (unflatten d i) |-> (acc s (unflatten d i)));
    kpost = (fun (i : natlt (sizeof d)) -> Cell a (unflatten d i) |-> f (acc s (unflatten d i)));
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn map_gpu
  (#et : Type0) (#r : erased nat) (#d : shape r) (cd : cshape d)
  (f : et -> et)
  (#l : tlayout d) {| ctlayout l |}
  (n : sz{SZ.v n == sizeof d /\ n <= max_blocks * max_threads})
  (a : tensor et l { is_global a })
  (#s : chest d et)
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> chest_map f s)
{
  launch_sync (kmap cd f n a);
}
