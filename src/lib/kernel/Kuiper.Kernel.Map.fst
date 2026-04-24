module Kuiper.Kernel.Map

(* Simple kernel: pointwise map of a function on an array. *)

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.Seq.Common

open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }

ghost
fn explode_setup
  (#et : Type0)
  (lena : nat)
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (#s : erased (lseq et lena))
  ()
  norewrite
  requires
    (a |-> s)
  ensures
    (forall+ (bid : natlt lena).
      Cell a bid |-> (Seq.index s bid)) **
    pure (SZ.fits (layout_size l))
{
  Array1.pts_to_ref a;
  Array1.explode a;
}

ghost
fn explode_teardown
  (#et : Type0)
  (f : et -> et)
  (lena : nat)
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (#s : erased (lseq et lena))
  ()
  norewrite
  requires
    (forall+ (bid : natlt lena).
      Cell a bid |-> (f (s @! bid))) **
    pure (SZ.fits (layout_size l))
  ensures
    a |-> (seq_map f s <: lseq et lena)
{
  forevery_map
    (fun (i:natlt lena) -> Cell a i |-> (f (s @! i)))
    (fun (i:natlt lena) -> Cell a i |-> ((seq_map f s)@!i))
    fn x { () };
  Array1.implode a;
}

inline_for_extraction noextract
fn kf_map
  (#et : Type0)
  (f : et -> et)
  (#lena : erased nat)
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (#s : erased (lseq et lena) )
  (bid : szlt lena)
  ()
  requires
    gpu **
    Cell a (bid <: natlt lena) |-> (s@!bid)
  ensures
    gpu **
    Cell a (bid <: natlt lena) |-> (f (s@!bid))
{
  let x = Array1.read_cell a bid;
  Array1.write_cell a bid (f x);
}

inline_for_extraction noextract
let kmap
  (#et : Type0)
  (f: et -> et)
  (lena : szp{ lena <= max_blocks * max_threads})
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (#_ : squash (Array1.is_global a))
  (#s : erased (lseq et lena))
  : kernel_desc
      (requires a |-> s)
      (ensures  a |-> lseq_map f s)
= {
    nthr = lena;
    f = kf_map f a;

    frame    = pure (SZ.fits (layout_size l));
    teardown = explode_teardown f lena a;
    setup    = explode_setup lena a;
    kpre =  (fun (i:natlt lena) -> Cell a i |-> (s@!i));
    kpost = (fun (i:natlt lena) -> Cell a i |-> (f (s@!i)));
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn map_gpu
  (#et : Type0)
  (f: et -> et)
  (lena : szp{ lena <= max_blocks * max_threads})
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (#_ : squash (Array1.is_global a))
  (#s: erased (lseq et lena))
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> lseq_map f s)
{
  launch_sync (kmap f lena a);
}

inline_for_extraction noextract
fn map_host
  (#et : Type0) {| sized et |}
  (f: et -> et)
  (lena : szp{ lena <= max_blocks * max_threads})
  (a : Pulse.Lib.Vec.lvec et lena)
  (#s: erased (lseq et lena))
  preserves cpu
  requires  a |-> s
  ensures   a |-> lseq_map f s
{
  let ga = Array1.alloc0 #et lena (l1_forward _);
  Array1.memcpy_host_to_device ga a lena;
  map_gpu f lena ga;
  Array1.memcpy_device_to_host' a 0sz ga 0sz lena;
  Array1.free ga;
  with s'. assert a |-> s';
  assert pure (Seq.equal s' (lseq_map f s));
  ();
}
