module Kuiper.Kernel.TMap

(* Simple kernel: pointwise map of a function on a tensor. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Shareable
module SZ = Kuiper.SizeT

ghost
fn setup
  (#et : Type0) (#r : nat) (#d : shape r)
  (n : szp{SZ.v n == sizeof d /\ n <= max_blocks * max_threads}) // sigh
  (frame : perm -> slprop) {| shareable frame |}
  (vf : abs d -> et -> et -> prop) // spec for f
  (#l : tlayout d)
  (a : tensor et l)
  (#s : chest d et)
  (#fr: perm)
  ()
  norewrite
  requires
    frame fr ** (a |-> s)
  ensures
    (forall+ (i : natlt n).
      frame (fr /. n) ** Cell a (unflatten d i) |-> acc s (unflatten d i)) **
    pure (SZ.fits (tlayout_ulen l))
{
  tensor_pts_to_ref a;
  tensor_explode a;
  forevery_iso (flatten_bij d) (fun (i : abs d) -> Cell a i |-> acc s i);
  forevery_rw_size (sizeof d) (SZ.v n);
  share_n frame n;
  forevery_zip (fun (_ : natlt n) -> frame (fr /. n)) _
}

ghost
fn teardown
  (#et : Type0) (#r : nat) (#d : shape r)
  (n : szp{SZ.v n == sizeof d /\ n <= max_blocks * max_threads}) // sigh
  (frame : perm -> slprop) {| shareable frame |}
  (vf : abs d -> et -> et -> prop) // spec for f
  (#l : tlayout d)
  (a : tensor et l)
  (#s : chest d et)
  (#fr: perm)
  ()
  norewrite
  requires
    (forall+ (i : natlt n).
      frame (fr /. n) **
      exists* (v : et).
        Cell a (unflatten d i) |-> v **
        pure (vf (unflatten d i) (acc s (unflatten d i)) v)
    ) **
    pure (SZ.fits (tlayout_ulen l))
  ensures
    frame fr **
    (exists* s'.
      a |-> s' **
      pure (chest_foralli (fun i x -> vf i (acc s i) x) s'))
{
  forevery_unzip _ _;
  gather_n frame n;
  let accs = forevery_exists #(natlt n) _;
  forevery_unzip _ _;
  forevery_elim_pure _;
  forevery_rw_size (SZ.v n) (sizeof d);
  assert forall+ (i: natlt (sizeof d)). tensor_pts_to_cell a (unflatten d i) (accs i);
  forevery_ext #(natlt (sizeof d))
    (fun i -> tensor_pts_to_cell a (unflatten d i) (accs i))
    (fun i -> tensor_pts_to_cell a (unflatten d i) (accs (flatten d (unflatten d i))));
  forevery_iso_back (flatten_bij d) (fun (i : abs d) -> tensor_pts_to_cell a i (accs (flatten d i)));
  let s' = Kuiper.Chest.mk d (fun i -> accs (flatten d i));
  forevery_map
    (fun (i : abs d) -> Cell a i |-> (accs (flatten d i)))
    (fun (i : abs d) -> Cell a i |-> (s' `acc` i))
    fn x { () };
  tensor_implode a;
  ()
}

inline_for_extraction noextract
fn kf
  (#et : Type0) (#r : erased nat) (#d : shape r) (cd : cshape d)
  (frame : perm -> slprop) {| shareable frame |}
  (vf : abs d -> et -> et -> prop) // spec for f
  (f :
    fn (#fr: perm) (i : conc d) (x : et)
      preserves frame fr
      returns r : et
      ensures pure (vf (up i) x r))
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (fr: perm)
  (#s : chest d et)
  (i : szlt (sizeof d))
  ()
  requires
    gpu **
    frame fr **
    Cell a (unflatten d i) |-> acc s (unflatten d i)
  ensures
    gpu **
    frame fr **
    (exists* (v : et).
      Cell a (unflatten d i) |-> v **
      pure (vf (unflatten d i) (acc s (unflatten d i)) v))
{
  rewrite
    Cell a (unflatten d i) |-> acc s (unflatten d i)
  as
    Cell a (up (cunflatten cd i)) |-> acc s (unflatten d i);
  let x = tensor_read_cell a (cunflatten cd i);
  tensor_write_cell a (cunflatten cd i) (f (cunflatten cd i) x);
  with v. assert tensor_pts_to_cell a (up (cunflatten cd i)) v;
  rewrite
    Cell a (up (cunflatten cd i)) |-> v
  as
    Cell a (unflatten d i) |-> v;
  ()
}

inline_for_extraction noextract
let kmap
  (#et : Type0) (#r : erased nat) (#d : shape r) (cd : cshape d)
  (frame : perm -> slprop) {| shareable frame |}
  (vf : abs d -> et -> et -> prop) // spec for f
  (f :
    fn (#fr: perm) (i : conc d) (x : et)
      preserves frame fr
      returns r : et
      ensures pure (vf (up i) x r))
  (#l : tlayout d) {| ctlayout l |}
  (n : szp{SZ.v n == sizeof d /\ n <= max_blocks * max_threads})
  (a : tensor et l)
  (#s : chest d et)
  (#_ : is_global a)
  (#fr: perm)
  : kernel_desc
      (requires frame fr ** a |-> s)
      (ensures  frame fr ** exists* s'. a |-> s' **
        pure (chest_foralli (fun i x -> vf i (acc s i) x) s'))
= {
    nthr = n;
    f = kf cd frame vf f a (fr /. n) #s;

    frame    = pure (SZ.fits (tlayout_ulen l));
    setup    = setup    n frame vf a #s;
    teardown = teardown n frame vf a #s;
    kpre  = (fun (i : natlt (sizeof d)) -> frame (fr /. n) ** Cell a (unflatten d i) |-> (acc s (unflatten d i)));
    kpost = (fun (i : natlt (sizeof d)) -> frame (fr /. n) ** exists* v. tensor_pts_to_cell a (unflatten d i) v **
                                             pure (vf (unflatten d i) (acc s (unflatten d i)) v));
    kpost_sendable = magic (); // LATER: frame needs to be sendable
    kpre_sendable  = magic ();
  } <: kernel_desc_n _ _

let vf_equal
  (#et : Type)
  (#r : erased nat) (#d : shape r)
  (f : et -> et)
 : (abs d -> et -> et -> prop) = (fun (_ : abs d) (x : et) (r : et) -> r == f x)

inline_for_extraction noextract
fn ff_from_pure u#a
  (#et : Type u#a)
  (#r : erased nat) (#d : shape r)
  (f : et -> et)
  (#fr: perm) (i : conc d) (x : et)
  norewrite
  preserves emp
  returns r : et
  ensures pure (vf_equal f (up i) x r)
{
  f x;
}

inline_for_extraction noextract
fn map_gpu
  (#et : Type0) (#r : erased nat) (#d : shape r) (cd : cshape d)
  (f : et -> et)
  (#l : tlayout d) {| ctlayout l |}
  (n : szp{SZ.v n == sizeof d /\ n <= max_blocks * max_threads})
  (a : tensor et l { is_global a })
  (#s : chest d et)
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> chest_map f s)
{
  launch_sync (kmap cd (fun _ -> emp) #(emp_shareable) (vf_equal f) (ff_from_pure f) n a #s #_ #1.0R);
  with s'. assert on gpu_loc (a |-> s');
  assert pure (Kuiper.Chest.equal s' (chest_map f s));
  ()
}
