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

(* In-place map with index: a[i] := f a[i] i.  The index is passed as a
   runtime SZ.t value (the only kind of natural-like index F* can pass to
   a stateful body without ghostness). *)

ghost
fn explode_teardown_mapi
  (#et : Type0)
  (lena : nat { SZ.fits lena })
  (f : et -> (i:SZ.t { SZ.v i < lena }) -> et)
  (#l : Array1.layout lena)
  (a : Array1.t et l)
  (#s : erased (lseq et lena))
  ()
  norewrite
  requires
    (forall+ (bid : natlt lena).
      Cell a bid |-> (f (s @! bid) (SZ.uint_to_t bid))) **
    pure (SZ.fits (layout_size l))
  ensures
    a |-> (lseq_mapi f s <: lseq et lena)
{
  forevery_map
    (fun (i:natlt lena) -> Cell a i |-> (f (s @! i) (SZ.uint_to_t i)))
    (fun (i:natlt lena) -> Cell a i |-> ((lseq_mapi f s) @! i))
    fn x { () };
  Array1.implode a;
}

inline_for_extraction noextract
fn kf_mapi
  (#et : Type0)
  (#lena : erased nat { SZ.fits lena })
  (f : et -> (i:SZ.t { SZ.v i < lena }) -> et)
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (#s : erased (lseq et lena))
  (bid : szlt lena)
  ()
  requires
    gpu **
    Cell a (bid <: natlt lena) |-> (s @! bid)
  ensures
    gpu **
    Cell a (bid <: natlt lena) |-> (f (s @! bid) (SZ.uint_to_t (bid <: natlt lena)))
{
  let x = Array1.read_cell a bid;
  Array1.write_cell a bid (f x bid);
}

inline_for_extraction noextract
let kmapi
  (#et : Type0)
  (lena : szp { lena <= max_blocks * max_threads })
  (f : et -> (i:SZ.t { SZ.v i < lena }) -> et)
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l)
  (#_ : squash (Array1.is_global a))
  (#s : erased (lseq et lena))
  : kernel_desc
      (requires a |-> s)
      (ensures  a |-> lseq_mapi f s)
= {
    nthr = lena;
    f = kf_mapi #_ #(hide (SZ.v lena)) f a;

    frame    = pure (SZ.fits (layout_size l));
    teardown = explode_teardown_mapi lena f a;
    setup    = explode_setup lena a;
    kpre =  (fun (i:natlt lena) -> Cell a i |-> (s @! i));
    kpost = (fun (i:natlt lena) -> Cell a i |-> (f (s @! i) (SZ.uint_to_t i)));
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn mapi_gpu
  (#et : Type0)
  (lena : szp { lena <= max_blocks * max_threads })
  (f : et -> (i:SZ.t { SZ.v i < lena }) -> et)
  (#l : Array1.layout lena) {| ctlayout l |}
  (a : Array1.t et l { Array1.is_global a })
  (#s : erased (lseq et lena))
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> lseq_mapi f s)
{
  launch_sync (kmapi lena f a);
}

(* Pointwise map from one array to another with possibly different element types.
   c[i] := f a[i]. *)

ghost
fn explode_setup_notinplace
  (#et #ot : Type0)
  (lena : szp)
  (#la : Array1.layout lena) (#lc : Array1.layout lena)
  (a : Array1.t et la)
  (c : Array1.t ot lc)
  (#sa : erased (lseq et lena))
  (#sc : erased (lseq ot lena))
  (#fa : perm)
  ()
  norewrite
  requires
    (a |-> Frac fa sa) ** (c |-> sc)
  ensures
    (forall+ (i : natlt lena).
      a |-> Frac (fa /. lena) sa **
      Cell c i |-> (sc @! i)) **
    pure (SZ.fits (Array1.layout_size lc))
{
  Array1.share_n a lena;
  Array1.pts_to_ref c;
  Array1.explode c;
  forevery_zip
    (fun (_ : natlt lena) -> a |-> Frac (fa /. lena) sa)
    (fun (i : natlt lena) -> Cell c i |-> (sc @! i));
  ()
}

ghost
fn explode_teardown_notinplace
  (#et #ot : Type0)
  (f : et -> ot)
  (lena : szp)
  (#la : Array1.layout lena) (#lc : Array1.layout lena)
  (a : Array1.t et la)
  (c : Array1.t ot lc)
  (#sa : erased (lseq et lena))
  (#fa : perm)
  ()
  norewrite
  requires
    (forall+ (i : natlt lena).
      a |-> Frac (fa /. lena) sa **
      Cell c i |-> (f (sa @! i))) **
    pure (SZ.fits (Array1.layout_size lc))
  ensures
    (a |-> Frac fa sa) **
    (c |-> (lseq_map f sa <: lseq ot lena))
{
  forevery_unzip
    (fun (_ : natlt lena) -> a |-> Frac (fa /. lena) sa)
    (fun (i : natlt lena) -> Cell c i |-> (f (sa @! i)));
  Array1.gather_n a lena;
  forevery_map
    (fun (i : natlt lena) -> Cell c i |-> (f (sa @! i)))
    (fun (i : natlt lena) -> Cell c i |-> ((lseq_map f sa) @! i))
    fn x { () };
  Array1.implode c;
  ()
}

inline_for_extraction noextract
fn kf_map_notinplace
  (#et #ot : Type0)
  (f : et -> ot)
  (#lena : erased nat)
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lc : Array1.layout lena) {| ctlayout lc |}
  (a : Array1.t et la)
  (c : Array1.t ot lc)
  (#sa : erased (lseq et lena))
  (#sc : erased (lseq ot lena))
  (#fa : perm)
  (i : szlt lena)
  ()
  requires
    gpu **
    a |-> Frac fa sa **
    Cell c (i <: natlt lena) |-> (sc @! i)
  ensures
    gpu **
    a |-> Frac fa sa **
    Cell c (i <: natlt lena) |-> (f (sa @! i))
{
  let x = Array1.read a i;
  Array1.write_cell c i (f x);
}

inline_for_extraction noextract
let kmap_notinplace
  (#et #ot : Type0)
  (f : et -> ot)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lc : Array1.layout lena) {| ctlayout lc |}
  (a : Array1.t et la)
  (c : Array1.t ot lc)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array1.is_global c))
  (#sa : erased (lseq et lena))
  (#sc : erased (lseq ot lena))
  (#fa : perm)
  : kernel_desc
      (requires (a |-> Frac fa sa) ** (c |-> sc))
      (ensures  (a |-> Frac fa sa) ** (c |-> (lseq_map f sa <: lseq ot lena)))
= {
    nthr = lena;
    f = kf_map_notinplace f a c;

    frame    = pure (SZ.fits (Array1.layout_size lc));
    teardown = explode_teardown_notinplace f lena a c;
    setup    = explode_setup_notinplace lena a c;
    kpre  = (fun (i : natlt lena) ->
      a |-> Frac (fa /. lena) sa ** Cell c i |-> (sc @! i));
    kpost = (fun (i : natlt lena) ->
      a |-> Frac (fa /. lena) sa ** Cell c i |-> (f (sa @! i)));
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn map_gpu_notinplace
  (#et #ot : Type0)
  (f : et -> ot)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lc : Array1.layout lena) {| ctlayout lc |}
  (a : Array1.t et la)
  (c : Array1.t ot lc)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array1.is_global c))
  (#sa : erased (lseq et lena))
  (#sc : erased (lseq ot lena))
  (#fa : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa)
  requires  on gpu_loc (c |-> sc)
  ensures   on gpu_loc (c |-> (lseq_map f sa <: lseq ot lena))
{
  launch_sync (kmap_notinplace f lena a c);
}

(* Two-array elementwise map: a[i] := f a[i] b[i]. *)

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

(* Three-array elementwise map into a separate output array, each array with a
   possibly different element type: o[i] := f a[i] b[i] c[i].
   The inputs a, b, c are held read-only; o receives the result. *)

ghost
fn explode_setup_3
  (#eta #etb #etc #eto : Type0)
  (lena : szp)
  (#la : Array1.layout lena) (#lb : Array1.layout lena)
  (#lc : Array1.layout lena) (#lo : Array1.layout lena)
  (a : Array1.t eta la)
  (b : Array1.t etb lb)
  (c : Array1.t etc lc)
  (o : Array1.t eto lo)
  (#sa : erased (lseq eta lena))
  (#sb : erased (lseq etb lena))
  (#sc : erased (lseq etc lena))
  (#so : erased (lseq eto lena))
  (#fa #fb #fc : perm)
  ()
  norewrite
  requires
    (a |-> Frac fa sa) ** (b |-> Frac fb sb) ** (c |-> Frac fc sc) ** (o |-> so)
  ensures
    (forall+ (i : natlt lena).
      a |-> Frac (fa /. lena) sa **
      b |-> Frac (fb /. lena) sb **
      c |-> Frac (fc /. lena) sc **
      Cell o i |-> (so @! i)) **
    pure (SZ.fits (Array1.layout_size lo))
{
  Array1.share_n a lena;
  Array1.share_n b lena;
  Array1.share_n c lena;
  Array1.pts_to_ref o;
  Array1.explode o;
  forevery_zip
    (fun (_ : natlt lena) -> c |-> Frac (fc /. lena) sc)
    (fun (i : natlt lena) -> Cell o i |-> (so @! i));
  forevery_zip
    (fun (_ : natlt lena) -> b |-> Frac (fb /. lena) sb)
    (fun (i : natlt lena) ->
      c |-> Frac (fc /. lena) sc **
      Cell o i |-> (so @! i));
  forevery_zip
    (fun (_ : natlt lena) -> a |-> Frac (fa /. lena) sa)
    (fun (i : natlt lena) ->
      b |-> Frac (fb /. lena) sb **
      c |-> Frac (fc /. lena) sc **
      Cell o i |-> (so @! i));
  ()
}

ghost
fn explode_teardown_3
  (#eta #etb #etc #eto : Type0)
  (f : eta -> etb -> etc -> eto)
  (lena : szp)
  (#la : Array1.layout lena) (#lb : Array1.layout lena)
  (#lc : Array1.layout lena) (#lo : Array1.layout lena)
  (a : Array1.t eta la)
  (b : Array1.t etb lb)
  (c : Array1.t etc lc)
  (o : Array1.t eto lo)
  (#sa : erased (lseq eta lena))
  (#sb : erased (lseq etb lena))
  (#sc : erased (lseq etc lena))
  (#fa #fb #fc : perm)
  ()
  norewrite
  requires
    (forall+ (i : natlt lena).
      a |-> Frac (fa /. lena) sa **
      b |-> Frac (fb /. lena) sb **
      c |-> Frac (fc /. lena) sc **
      Cell o i |-> (f (sa @! i) (sb @! i) (sc @! i))) **
    pure (SZ.fits (Array1.layout_size lo))
  ensures
    (a |-> Frac fa sa) ** (b |-> Frac fb sb) ** (c |-> Frac fc sc) **
    (o |-> (lseq_map3 f sa sb sc <: lseq eto lena))
{
  forevery_unzip
    (fun (_ : natlt lena) -> a |-> Frac (fa /. lena) sa)
    (fun (i : natlt lena) ->
      b |-> Frac (fb /. lena) sb **
      c |-> Frac (fc /. lena) sc **
      Cell o i |-> (f (sa @! i) (sb @! i) (sc @! i)));
  forevery_unzip
    (fun (_ : natlt lena) -> b |-> Frac (fb /. lena) sb)
    (fun (i : natlt lena) ->
      c |-> Frac (fc /. lena) sc **
      Cell o i |-> (f (sa @! i) (sb @! i) (sc @! i)));
  forevery_unzip
    (fun (_ : natlt lena) -> c |-> Frac (fc /. lena) sc)
    (fun (i : natlt lena) -> Cell o i |-> (f (sa @! i) (sb @! i) (sc @! i)));
  Array1.gather_n a lena;
  Array1.gather_n b lena;
  Array1.gather_n c lena;
  forevery_map
    (fun (i : natlt lena) -> Cell o i |-> (f (sa @! i) (sb @! i) (sc @! i)))
    (fun (i : natlt lena) -> Cell o i |-> ((lseq_map3 f sa sb sc) @! i))
    fn x { () };
  Array1.implode o;
  ()
}

inline_for_extraction noextract
fn kf_map3
  (#eta #etb #etc #eto : Type0)
  (f : eta -> etb -> etc -> eto)
  (#lena : erased nat)
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lb : Array1.layout lena) {| ctlayout lb |}
  (#lc : Array1.layout lena) {| ctlayout lc |}
  (#lo : Array1.layout lena) {| ctlayout lo |}
  (a : Array1.t eta la)
  (b : Array1.t etb lb)
  (c : Array1.t etc lc)
  (o : Array1.t eto lo)
  (#sa : erased (lseq eta lena))
  (#sb : erased (lseq etb lena))
  (#sc : erased (lseq etc lena))
  (#so : erased (lseq eto lena))
  (#fa #fb #fc : perm)
  (i : szlt lena)
  ()
  requires
    gpu **
    a |-> Frac fa sa **
    b |-> Frac fb sb **
    c |-> Frac fc sc **
    Cell o (i <: natlt lena) |-> (so @! i)
  ensures
    gpu **
    a |-> Frac fa sa **
    b |-> Frac fb sb **
    c |-> Frac fc sc **
    Cell o (i <: natlt lena) |-> (f (sa @! i) (sb @! i) (sc @! i))
{
  let x = Array1.read a i;
  let y = Array1.read b i;
  let z = Array1.read c i;
  Array1.write_cell o i (f x y z);
}

inline_for_extraction noextract
let kmap3
  (#eta #etb #etc #eto : Type0)
  (f : eta -> etb -> etc -> eto)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lb : Array1.layout lena) {| ctlayout lb |}
  (#lc : Array1.layout lena) {| ctlayout lc |}
  (#lo : Array1.layout lena) {| ctlayout lo |}
  (a : Array1.t eta la)
  (b : Array1.t etb lb)
  (c : Array1.t etc lc)
  (o : Array1.t eto lo)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array1.is_global b))
  (#_ : squash (Array1.is_global c))
  (#_ : squash (Array1.is_global o))
  (#sa : erased (lseq eta lena))
  (#sb : erased (lseq etb lena))
  (#sc : erased (lseq etc lena))
  (#so : erased (lseq eto lena))
  (#fa #fb #fc : perm)
  : kernel_desc
      (requires (a |-> Frac fa sa) ** (b |-> Frac fb sb) ** (c |-> Frac fc sc) ** (o |-> so))
      (ensures  (a |-> Frac fa sa) ** (b |-> Frac fb sb) ** (c |-> Frac fc sc) **
                (o |-> (lseq_map3 f sa sb sc <: lseq eto lena)))
= {
    nthr = lena;
    f = kf_map3 f a b c o;

    frame    = pure (SZ.fits (Array1.layout_size lo));
    teardown = explode_teardown_3 f lena a b c o;
    setup    = explode_setup_3 lena a b c o;
    kpre  = (fun (i : natlt lena) ->
      a |-> Frac (fa /. lena) sa **
      b |-> Frac (fb /. lena) sb **
      c |-> Frac (fc /. lena) sc **
      Cell o i |-> (so @! i));
    kpost = (fun (i : natlt lena) ->
      a |-> Frac (fa /. lena) sa **
      b |-> Frac (fb /. lena) sb **
      c |-> Frac (fc /. lena) sc **
      Cell o i |-> (f (sa @! i) (sb @! i) (sc @! i)));
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn map_gpu3
  (#eta #etb #etc #eto : Type0)
  (f : eta -> etb -> etc -> eto)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : Array1.layout lena) {| ctlayout la |}
  (#lb : Array1.layout lena) {| ctlayout lb |}
  (#lc : Array1.layout lena) {| ctlayout lc |}
  (#lo : Array1.layout lena) {| ctlayout lo |}
  (a : Array1.t eta la)
  (b : Array1.t etb lb)
  (c : Array1.t etc lc)
  (o : Array1.t eto lo)
  (#_ : squash (Array1.is_global a))
  (#_ : squash (Array1.is_global b))
  (#_ : squash (Array1.is_global c))
  (#_ : squash (Array1.is_global o))
  (#sa : erased (lseq eta lena))
  (#sb : erased (lseq etb lena))
  (#sc : erased (lseq etc lena))
  (#so : erased (lseq eto lena))
  (#fa #fb #fc : perm)
  norewrite
  preserves cpu ** on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb) ** on gpu_loc (c |-> Frac fc sc)
  requires  on gpu_loc (o |-> so)
  ensures   on gpu_loc (o |-> (lseq_map3 f sa sb sc <: lseq eto lena))
{
  launch_sync (kmap3 f lena a b c o);
}
