module Kuiper.Kernel.Cat

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.Seq.Common
open Kuiper.Tensor { ctlayout }
open Kuiper.Tensor.Layout.Alg { l1_forward }

ghost
fn explode_setup_cat2
  (#et : Type0)
  (lena : szp)
  (lenb : szp)
  (lenab : szp { SZ.v lenab == lena + lenb })
  (#la : Array1.layout lena)
  (#lb : Array1.layout lenb)
  (#lo : Array1.layout lenab)
  (a : Array1.t et la)
  (b : Array1.t et lb)
  (out : Array1.t et lo)
  (#sa : erased (lseq et lena))
  (#sb : erased (lseq et lenb))
  (#so : erased (lseq et lenab))
  (#fa #fb : perm)
  ()
  norewrite
  requires
    (a |-> Frac fa sa) ** (b |-> Frac fb sb) ** (out |-> so)
  ensures
    (forall+ (i : natlt lenab).
      a |-> Frac (fa /. lenab) sa **
      b |-> Frac (fb /. lenab) sb **
      Cell out i |-> (so @! i)) **
    pure (SZ.fits (Array1.layout_size lo))
{
  Array1.share_n a lenab;
  Array1.share_n b lenab;
  Array1.pts_to_ref out;
  Array1.explode out;
  forevery_zip
    (fun (_ : natlt lenab) -> b |-> Frac (fb /. lenab) sb)
    (fun (i : natlt lenab) -> Cell out i |-> (so @! i));
  forevery_zip
    (fun (_ : natlt lenab) -> a |-> Frac (fa /. lenab) sa)
    (fun (i : natlt lenab) ->
      b |-> Frac (fb /. lenab) sb ** Cell out i |-> (so @! i));
  ()
}

ghost
fn explode_teardown_cat2
  (#et : Type0)
  (lena : szp)
  (lenb : szp)
  (lenab : szp { SZ.v lenab == lena + lenb })
  (#la : Array1.layout lena)
  (#lb : Array1.layout lenb)
  (#lo : Array1.layout lenab)
  (a : Array1.t et la)
  (b : Array1.t et lb)
  (out : Array1.t et lo)
  (#sa : erased (lseq et lena))
  (#sb : erased (lseq et lenb))
  (#fa #fb : perm)
  ()
  norewrite
  requires
    (forall+ (i : natlt lenab).
      a |-> Frac (fa /. lenab) sa **
      b |-> Frac (fb /. lenab) sb **
      Cell out i |-> ((cat2_seq sa sb <: lseq et lenab) @! i)) **
    pure (SZ.fits (Array1.layout_size lo))
  ensures
    (a |-> Frac fa sa) **
    (b |-> Frac fb sb) **
    (out |-> (cat2_seq sa sb <: lseq et lenab))
{
  forevery_unzip
    (fun (_ : natlt lenab) -> a |-> Frac (fa /. lenab) sa)
    _;
  Array1.gather_n a lenab;
  forevery_unzip
    (fun (_ : natlt lenab) -> b |-> Frac (fb /. lenab) sb)
    _;
  Array1.gather_n b lenab;
  Array1.implode out;
  ()
}

inline_for_extraction noextract
fn kf_cat2
  (#et : Type0) {| sized et |}
  (lena_sz : szp)
  (#lenb : erased nat)
  (lenab : szp { SZ.v lenab == lena_sz + lenb })
  (#la : Array1.layout lena_sz) {| ctlayout la |}
  (#lb : Array1.layout lenb) {| ctlayout lb |}
  (#lo : Array1.layout lenab) {| ctlayout lo |}
  (a : Array1.t et la)
  (b : Array1.t et lb)
  (out : Array1.t et lo)
  (#sa : erased (lseq et lena_sz))
  (#sb : erased (lseq et lenb))
  (#so : erased (lseq et lenab))
  (#fa #fb : perm)
  (i : szlt lenab)
  ()
  requires
    gpu **
    a |-> Frac fa sa **
    b |-> Frac fb sb **
    Cell out (i <: natlt lenab) |-> (so @! i)
  ensures
    gpu **
    a |-> Frac fa sa **
    b |-> Frac fb sb **
    Cell out (i <: natlt lenab) |-> ((cat2_seq sa sb <: lseq et lenab) @! i)
{
  if (i <^ lena_sz) {
    let x = Array1.read a i;
    Array1.write_cell out i x;
  } else {
    let j = i -^ lena_sz;
    let x = Array1.read b j;
    Array1.write_cell out i x;
  }
}

inline_for_extraction noextract
let kcat2
  (#et : Type0) {| sized et |}
  (lena : szp)
  (lenb : szp)
  (lenab : szp { SZ.v lenab == lena + lenb /\ lenab <= max_blocks * max_threads })
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lb : Array1.layout lenb) {| ctlayout lb |}
  (#lo : Array1.layout lenab) {| ctlayout lo |}
  (a : Array1.t et la)
  (b : Array1.t et lb)
  (out : Array1.t et lo)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array1.is_global b))
  (#_ : squash (Array1.is_global out))
  (#sa : erased (lseq et lena))
  (#sb : erased (lseq et lenb))
  (#so : erased (lseq et lenab))
  (#fa #fb : perm)
  : kernel_desc
      (requires (a |-> Frac fa sa) ** (b |-> Frac fb sb) ** (out |-> so))
      (ensures
        (a |-> Frac fa sa) **
        (b |-> Frac fb sb) **
        (out |-> (cat2_seq sa sb <: lseq et lenab)))
= {
    nthr = lenab;
    f = kf_cat2 lena lenab a b out;

    frame    = pure (SZ.fits (Array1.layout_size lo));
    teardown = explode_teardown_cat2 lena lenb lenab a b out;
    setup    = explode_setup_cat2 lena lenb lenab a b out;
    kpre  = (fun (i : natlt lenab) ->
      a |-> Frac (fa /. lenab) sa **
      b |-> Frac (fb /. lenab) sb **
      Cell out i |-> (so @! i));
    kpost = (fun (i : natlt lenab) ->
      a |-> Frac (fa /. lenab) sa **
      b |-> Frac (fb /. lenab) sb **
      Cell out i |-> ((cat2_seq sa sb <: lseq et lenab) @! i));
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn cat2_gpu
  (#et : Type0) {| sized et |}
  (lena : szp)
  (lenb : szp { SZ.fits (lena + lenb) /\ lena + lenb <= max_blocks * max_threads })
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lb : Array1.layout lenb) {| ctlayout lb |}
  (a : Array1.t et la { Array1.is_global a })
  (b : Array1.t et lb { Array1.is_global b })
  (#sa : erased (lseq et lena))
  (#sb : erased (lseq et lenb))
  (#fa #fb : perm)
  preserves cpu
  preserves on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb)
  returns out : Array1.t et (l1_forward (lena + lenb))
  ensures
    on gpu_loc (out |-> (cat2_seq sa sb <: lseq et (lena + lenb))) **
    pure (Array1.is_global out)
{
  let lenab : szp = lena +^ lenb;
  let out = Array1.alloc0 #et lenab (l1_forward lenab);
  launch_sync (kcat2 lena lenb lenab a b out);
  out
}
