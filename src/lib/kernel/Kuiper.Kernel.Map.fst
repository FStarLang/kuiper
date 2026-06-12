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
  (lena : nat { SZ.fits lena })
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
  (lena : nat { SZ.fits lena })
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
  (a : Array1.t et l { Array1.is_global a })
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

ghost
fn explode_setup_2
  (#et : Type0)
  (lena : szp)
  (#la : Array1.layout lena) (#lb : Array1.layout lena)
  (a : Array1.t et la)
  (b : Array1.t et lb)
  (#sa : erased (lseq et lena))
  (#sb : erased (lseq et lena))
  (#fb : perm)
  ()
  norewrite
  requires
    (a |-> sa) ** (b |-> Frac fb sb)
  ensures
    (forall+ (i : natlt lena).
      Cell a i |-> (sa @! i) **
      b |-> Frac (fb /. lena) sb) **
    pure (SZ.fits (Array1.layout_size la))
{
  Array1.pts_to_ref a;
  Array1.share_n b lena;
  Array1.explode a;
  forevery_zip
    (fun (i : natlt lena) -> Cell a i |-> (sa @! i))
    (fun (_ : natlt lena) -> b |-> Frac (fb /. lena) sb);
  ()
}

ghost
fn explode_teardown_2
  (#et : Type0)
  (f : et -> et -> et)
  (lena : szp)
  (#la : Array1.layout lena) (#lb : Array1.layout lena)
  (a : Array1.t et la)
  (b : Array1.t et lb)
  (#sa : erased (lseq et lena))
  (#sb : erased (lseq et lena))
  (#fb : perm)
  ()
  norewrite
  requires
    (forall+ (i : natlt lena).
      Cell a i |-> (f (sa @! i) (sb @! i)) **
      b |-> Frac (fb /. lena) sb) **
    pure (SZ.fits (Array1.layout_size la))
  ensures
    (a |-> (lseq_map2 f sa sb <: lseq et lena)) **
    (b |-> Frac fb sb)
{
  forevery_unzip
    (fun (i : natlt lena) -> Cell a i |-> (f (sa @! i) (sb @! i)))
    (fun (_ : natlt lena) -> b |-> Frac (fb /. lena) sb);
  Array1.gather_n b lena;
  forevery_map
    (fun (i : natlt lena) -> Cell a i |-> (f (sa @! i) (sb @! i)))
    (fun (i : natlt lena) -> Cell a i |-> ((lseq_map2 f sa sb) @! i))
    fn x { () };
  Array1.implode a;
  ()
}

inline_for_extraction noextract
fn kf_map2
  (#et : Type0)
  (f : et -> et -> et)
  (#lena : erased nat)
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lb : Array1.layout lena) {| ctlayout lb |}
  (a : Array1.t et la)
  (b : Array1.t et lb)
  (#sa : erased (lseq et lena))
  (#sb : erased (lseq et lena))
  (#fb : perm)
  (i : szlt lena)
  ()
  requires
    gpu **
    Cell a (i <: natlt lena) |-> (sa @! i) **
    b |-> Frac fb sb
  ensures
    gpu **
    Cell a (i <: natlt lena) |-> (f (sa @! i) (sb @! i)) **
    b |-> Frac fb sb
{
  let x = Array1.read_cell a i;
  let y = Array1.read b i;
  Array1.write_cell a i (f x y);
}

inline_for_extraction noextract
let kmap2
  (#et : Type0)
  (f : et -> et -> et)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lb : Array1.layout lena) {| ctlayout lb |}
  (a : Array1.t et la)
  (b : Array1.t et lb)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array1.is_global b))
  (#sa : erased (lseq et lena))
  (#sb : erased (lseq et lena))
  (#fb : perm)
  : kernel_desc
      (requires (a |-> sa) ** (b |-> Frac fb sb))
      (ensures  (a |-> (lseq_map2 f sa sb <: lseq et lena)) ** (b |-> Frac fb sb))
= {
    nthr = lena;
    f = kf_map2 f a b;

    frame    = pure (SZ.fits (Array1.layout_size la));
    teardown = explode_teardown_2 f lena a b;
    setup    = explode_setup_2 lena a b;
    kpre  = (fun (i : natlt lena) ->
      Cell a i |-> (sa @! i) ** b |-> Frac (fb /. lena) sb);
    kpost = (fun (i : natlt lena) ->
      Cell a i |-> (f (sa @! i) (sb @! i)) ** b |-> Frac (fb /. lena) sb);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn map_gpu2
  (#et : Type0)
  (f : et -> et -> et)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lb : Array1.layout lena) {| ctlayout lb |}
  (a : Array1.t et la)
  (b : Array1.t et lb)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array1.is_global b))
  (#sa : erased (lseq et lena))
  (#sb : erased (lseq et lena))
  (#fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (b |-> Frac fb sb)
  requires  on gpu_loc (a |-> sa)
  ensures   on gpu_loc (a |-> (lseq_map2 f sa sb <: lseq et lena))
{
  launch_sync (kmap2 f lena a b);
}

ghost
fn explode_setup_gather
  (#et : Type0)
  (lens : szp)
  (leni : szp)
  (#ls : Array1.layout lens) (#li : Array1.layout leni) (#lo : Array1.layout leni)
  (src : Array1.t et ls)
  (idx : Array1.t u32 li)
  (out : Array1.t et lo)
  (#ss : erased (lseq et lens))
  (#si : erased (lseq u32 leni))
  (#so : erased (lseq et leni))
  (#fs #fi : perm)
  ()
  norewrite
  requires
    (src |-> Frac fs ss) ** (idx |-> Frac fi si) ** (out |-> so)
  ensures
    (forall+ (i : natlt leni).
      idx |-> Frac (fi /. leni) si **
      src |-> Frac (fs /. leni) ss **
      Cell out i |-> (so @! i)) **
    pure (SZ.fits (Array1.layout_size lo))
{
  Array1.share_n src leni;
  Array1.share_n idx leni;
  Array1.pts_to_ref out;
  Array1.explode out;
  forevery_zip
    (fun (_ : natlt leni) -> src |-> Frac (fs /. leni) ss)
    (fun (i : natlt leni) -> Cell out i |-> (so @! i));
  forevery_zip
    (fun (_ : natlt leni) -> idx |-> Frac (fi /. leni) si)
    (fun (i : natlt leni) ->
      src |-> Frac (fs /. leni) ss ** Cell out i |-> (so @! i));
  ()
}

ghost
fn explode_teardown_gather
  (#et : Type0)
  (lens : szp)
  (leni : szp)
  (#ls : Array1.layout lens) (#li : Array1.layout leni) (#lo : Array1.layout leni)
  (src : Array1.t et ls)
  (idx : Array1.t u32 li)
  (out : Array1.t et lo)
  (#ss : erased (lseq et lens))
  (#si : erased (lseq u32 leni))
  (#fs #fi : perm)
  ()
  norewrite
  requires
    (forall+ (i : natlt leni).
      idx |-> Frac (fi /. leni) si **
      src |-> Frac (fs /. leni) ss **
      Cell out i |-> (ss @! (FStar.UInt32.v (si @! i) % lens))) **
    pure (SZ.fits (Array1.layout_size lo))
  ensures
    (src |-> Frac fs ss) **
    (idx |-> Frac fi si) **
    (exists* so'. out |-> so')
{
  forevery_unzip
    (fun (_ : natlt leni) -> idx |-> Frac (fi /. leni) si)
    _;
  Array1.gather_n idx leni;
  forevery_unzip
    (fun (_ : natlt leni) -> src |-> Frac (fs /. leni) ss)
    _;
  Array1.gather_n src leni;
  forevery_map
    (fun (i : natlt leni) ->
      Cell out i |-> (ss @! (FStar.UInt32.v (si @! i) % lens)))
    (fun (i : natlt leni) ->
      Cell out i |-> (Seq.init_ghost leni (fun j -> ss @! (FStar.UInt32.v (si @! j) % lens)) @! i))
    fn x { () };
  Array1.implode out;
  ()
}

inline_for_extraction noextract
fn kf_gather
  (#et : Type0)
  (lens : szp)
  (#leni : erased nat)
  (#ls : Array1.layout lens) {| ctlayout ls |}
  (#li : Array1.layout leni) {| ctlayout li |}
  (#lo : Array1.layout leni) {| ctlayout lo |}
  (src : Array1.t et ls)
  (idx : Array1.t u32 li)
  (out : Array1.t et lo)
  (#ss : erased (lseq et lens))
  (#si : erased (lseq u32 leni))
  (#so : erased (lseq et leni))
  (#fs #fi : perm)
  (i : szlt leni)
  ()
  requires
    gpu **
    idx |-> Frac fi si **
    src |-> Frac fs ss **
    Cell out (i <: natlt leni) |-> (so @! i)
  ensures
    gpu **
    idx |-> Frac fi si **
    src |-> Frac fs ss **
    Cell out (i <: natlt leni) |-> (ss @! (FStar.UInt32.v (si @! i) % lens))
{
  let ix = Array1.read idx i;
  let j = SZ.uint32_to_sizet ix;
  let jm = j %^ lens;
  let x = Array1.read src jm;
  Array1.write_cell out i x;
}

inline_for_extraction noextract
let kgather
  (#et : Type0)
  (lens : szp)
  (leni : szp { leni <= max_blocks * max_threads })
  (#ls : Array1.layout lens) {| ctlayout ls |}
  (#li : Array1.layout leni) {| ctlayout li |}
  (#lo : Array1.layout leni) {| ctlayout lo |}
  (src : Array1.t et ls)
  (idx : Array1.t u32 li)
  (out : Array1.t et lo)
  (#_ : squash (Array1.is_global src))
  (#_ : squash (Array1.is_global idx))
  (#_ : squash (Array1.is_global out))
  (#ss : erased (lseq et lens))
  (#si : erased (lseq u32 leni))
  (#so : erased (lseq et leni))
  (#fs #fi : perm)
  : kernel_desc
      (requires (src |-> Frac fs ss) ** (idx |-> Frac fi si) ** (out |-> so))
      (ensures  (src |-> Frac fs ss) ** (idx |-> Frac fi si) ** (exists* so'. out |-> so'))
= {
    nthr = leni;
    f = kf_gather lens src idx out;

    frame    = pure (SZ.fits (Array1.layout_size lo));
    teardown = explode_teardown_gather lens leni src idx out;
    setup    = explode_setup_gather lens leni src idx out;
    kpre  = (fun (i : natlt leni) ->
      idx |-> Frac (fi /. leni) si **
      src |-> Frac (fs /. leni) ss **
      Cell out i |-> (so @! i));
    kpost = (fun (i : natlt leni) ->
      idx |-> Frac (fi /. leni) si **
      src |-> Frac (fs /. leni) ss **
      Cell out i |-> (ss @! (FStar.UInt32.v (si @! i) % lens)));
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn gather_gpu
  (#et : Type0)
  (lens : szp)
  (leni : szp { leni <= max_blocks * max_threads })
  (#ls : Array1.layout lens) {| ctlayout ls |}
  (#li : Array1.layout leni) {| ctlayout li |}
  (#lo : Array1.layout leni) {| ctlayout lo |}
  (src : Array1.t et ls)
  (idx : Array1.t u32 li)
  (out : Array1.t et lo)
  (#_ : squash (Array1.is_global src))
  (#_ : squash (Array1.is_global idx))
  (#_ : squash (Array1.is_global out))
  (#ss : erased (lseq et lens))
  (#si : erased (lseq u32 leni))
  (#so : erased (lseq et leni))
  (#fs #fi : perm)
  norewrite
  preserves cpu ** on gpu_loc (src |-> Frac fs ss) ** on gpu_loc (idx |-> Frac fi si)
  requires  on gpu_loc (out |-> so)
  ensures   exists* so'. on gpu_loc (out |-> so')
{
  launch_sync (kgather lens leni src idx out);
}
