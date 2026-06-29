module Kuiper.Kernel.Map

(* Simple kernel: pointwise map of a function on an array. *)

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
open Kuiper.Seq.Common
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Bijection { ( =~ ) }

(* Bijection between the abstract 1-D tensor index [(k, ())] and a plain
   [natlt len], used to (un)reindex a forevery over tensor cells. *)
let abs_bij (#len : nat) : (abs (len @| INil) =~ natlt len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
  }

ghost
fn explode_setup
  (#et : Type0)
  (lena : nat)
  (#l : layout1 lena)
  (a : array1 et l)
  (#s : erased (chest1 et lena))
  ()
  norewrite
  requires
    (a |-> s)
  ensures
    (forall+ (bid : natlt lena).
      Cell a (idx1 bid) |-> (acc1 s bid)) **
    pure (SZ.fits (tlayout_ulen l))
{
  tensor_pts_to_ref a;
  tensor_explode a;
  forevery_iso (abs_bij #lena)
    (fun (i : abs (lena @| INil)) -> Cell a i |-> (acc s i));
  forevery_ext
    (fun (y : natlt lena) -> Cell a (abs_bij.gg y) |-> (acc s (abs_bij.gg y)))
    (fun (bid : natlt lena) -> Cell a (idx1 bid) |-> (acc1 s bid));
}

ghost
fn explode_teardown
  (#et : Type0)
  (f : et -> et)
  (lena : nat)
  (#l : layout1 lena)
  (a : array1 et l)
  (#s : erased (chest1 et lena))
  ()
  norewrite
  requires
    (forall+ (bid : natlt lena).
      Cell a (idx1 bid) |-> (f (acc1 s bid))) **
    pure (SZ.fits (tlayout_ulen l))
  ensures
    a |-> chest_map f s
{
  forevery_ext
    (fun (bid : natlt lena) -> Cell a (idx1 bid) |-> (f (acc1 s bid)))
    (fun (y : natlt lena) -> Cell a (abs_bij.gg y) |-> (acc (chest_map f s) (abs_bij.gg y)));
  forevery_iso_back (abs_bij #lena)
    (fun (i : abs (lena @| INil)) -> Cell a i |-> (acc (chest_map f s) i));
  tensor_implode a;
}

inline_for_extraction noextract
fn kf_map
  (#et : Type0)
  (f : et -> et)
  (#lena : erased nat)
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l)
  (#s : erased (chest1 et lena) )
  (bid : szlt lena)
  ()
  requires
    gpu **
    Cell a (idx1 (bid <: natlt lena)) |-> (acc1 s bid)
  ensures
    gpu **
    Cell a (idx1 (bid <: natlt lena)) |-> (f (acc1 s bid))
{
  let x = tensor_read_cell a (cidx1 bid);
  tensor_write_cell a (cidx1 bid) (f x);
}

inline_for_extraction noextract
let kmap
  (#et : Type0)
  (f: et -> et)
  (lena : szp{ lena <= max_blocks * max_threads})
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l)
  (#_ : squash (is_global a))
  (#s : erased (chest1 et lena))
  : kernel_desc
      (requires a |-> s)
      (ensures  a |-> chest_map f s)
= {
    nthr = lena;
    f = kf_map f a;

    frame    = pure (SZ.fits (tlayout_ulen l));
    teardown = explode_teardown f lena a;
    setup    = explode_setup lena a;
    kpre =  (fun (i:natlt lena) -> Cell a (idx1 i) |-> (acc1 s i));
    kpost = (fun (i:natlt lena) -> Cell a (idx1 i) |-> (f (acc1 s i)));
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn map_gpu
  (#et : Type0)
  (f: et -> et)
  (lena : szp{ lena <= max_blocks * max_threads})
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#s: erased (chest1 et lena))
  preserves cpu
  requires  on gpu_loc (a |-> s)
  ensures   on gpu_loc (a |-> chest_map f s)
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
  let ga = alloc0 #et lena (l1_forward lena);
  with em. assert on gpu_loc (ga |-> em);

  (* Host -> device. *)
  map_loc gpu_loc
    #(ga |-> em)
    #(core ga |-> to_seq (l1_forward lena) em)
    fn _ { tensor_concr ga; };
  gpu_memcpy_host_to_device (core ga) a lena;
  map_loc gpu_loc
    #(core ga |-> reveal s)
    #(ga |-> from_seq (l1_forward lena) s)
    fn _ {
      tensor_abs' (l1_forward lena) (core ga);
      rewrite (from_array (l1_forward lena) (core ga) |-> from_seq (l1_forward lena) s)
           as (ga |-> from_seq (l1_forward lena) s);
    };

  map_gpu f lena ga;

  (* Device -> host. *)
  with res. assert on gpu_loc (ga |-> res);
  map_loc gpu_loc
    #(ga |-> res)
    #(core ga |-> to_seq (l1_forward lena) res)
    fn _ { tensor_concr ga; };
  gpu_memcpy_device_to_host a (core ga) lena;
  map_loc gpu_loc
    #(core ga |-> to_seq (l1_forward lena) res)
    #(ga |-> res)
    fn _ {
      tensor_abs (l1_forward lena) (core ga);
      rewrite (from_array (l1_forward lena) (core ga) |-> reveal res)
           as (ga |-> reveal res);
    };
  free ga;

  assert pure (Seq.equal (to_seq (l1_forward lena) res) (lseq_map f s));
  ();
}

ghost
fn explode_setup_2
  (#et : Type0)
  (lena : szp)
  (#la : layout1 lena) (#lb : layout1 lena)
  (a : array1 et la)
  (b : array1 et lb)
  (#sa : erased (chest1 et lena))
  (#sb : erased (chest1 et lena))
  (#fb : perm)
  ()
  norewrite
  requires
    (a |-> sa) ** (b |-> Frac fb sb)
  ensures
    (forall+ (i : natlt lena).
      Cell a (idx1 i) |-> (acc1 sa i) **
      b |-> Frac (fb /. lena) sb) **
    pure (SZ.fits (tlayout_ulen la))
{
  tensor_pts_to_ref a;
  tensor_share_n b lena;
  tensor_explode a;
  forevery_iso (abs_bij #lena)
    (fun (i : abs (lena @| INil)) -> Cell a i |-> (acc sa i));
  forevery_ext
    (fun (y : natlt lena) -> Cell a (abs_bij.gg y) |-> (acc sa (abs_bij.gg y)))
    (fun (i : natlt lena) -> Cell a (idx1 i) |-> (acc1 sa i));
  forevery_zip
    (fun (i : natlt lena) -> Cell a (idx1 i) |-> (acc1 sa i))
    (fun (_ : natlt lena) -> b |-> Frac (fb /. lena) sb);
  ()
}

ghost
fn explode_teardown_2
  (#et : Type0)
  (f : et -> et -> et)
  (lena : szp)
  (#la : layout1 lena) (#lb : layout1 lena)
  (a : array1 et la)
  (b : array1 et lb)
  (#sa : erased (chest1 et lena))
  (#sb : erased (chest1 et lena))
  (#fb : perm)
  ()
  norewrite
  requires
    (forall+ (i : natlt lena).
      Cell a (idx1 i) |-> (f (acc1 sa i) (acc1 sb i)) **
      b |-> Frac (fb /. lena) sb) **
    pure (SZ.fits (tlayout_ulen la))
  ensures
    (a |-> chest1_map2 f sa sb) **
    (b |-> Frac fb sb)
{
  forevery_unzip
    (fun (i : natlt lena) -> Cell a (idx1 i) |-> (f (acc1 sa i) (acc1 sb i)))
    (fun (_ : natlt lena) -> b |-> Frac (fb /. lena) sb);
  tensor_gather_n b lena;
  forevery_ext
    (fun (i : natlt lena) -> Cell a (idx1 i) |-> (f (acc1 sa i) (acc1 sb i)))
    (fun (y : natlt lena) -> Cell a (abs_bij.gg y) |-> (acc (chest1_map2 f sa sb) (abs_bij.gg y)));
  forevery_iso_back (abs_bij #lena)
    (fun (i : abs (lena @| INil)) -> Cell a i |-> (acc (chest1_map2 f sa sb) i));
  tensor_implode a;
  ()
}

inline_for_extraction noextract
fn kf_map2
  (#et : Type0)
  (f : et -> et -> et)
  (#lena : erased nat)
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (a : array1 et la)
  (b : array1 et lb)
  (#sa : erased (chest1 et lena))
  (#sb : erased (chest1 et lena))
  (#fb : perm)
  (i : szlt lena)
  ()
  requires
    gpu **
    Cell a (idx1 (i <: natlt lena)) |-> (acc1 sa i) **
    b |-> Frac fb sb
  ensures
    gpu **
    Cell a (idx1 (i <: natlt lena)) |-> (f (acc1 sa i) (acc1 sb i)) **
    b |-> Frac fb sb
{
  let x = tensor_read_cell a (cidx1 i);
  let y = tensor_read b (cidx1 i);
  tensor_write_cell a (cidx1 i) (f x y);
}

inline_for_extraction noextract
let kmap2
  (#et : Type0)
  (f : et -> et -> et)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (a : array1 et la)
  (b : array1 et lb)
  (#_ : squash (is_global a))
  (#_ : squash (is_global b))
  (#sa : erased (chest1 et lena))
  (#sb : erased (chest1 et lena))
  (#fb : perm)
  : kernel_desc
      (requires (a |-> sa) ** (b |-> Frac fb sb))
      (ensures  (a |-> chest1_map2 f sa sb) ** (b |-> Frac fb sb))
= {
    nthr = lena;
    f = kf_map2 f a b;

    frame    = pure (SZ.fits (tlayout_ulen la));
    teardown = explode_teardown_2 f lena a b;
    setup    = explode_setup_2 lena a b;
    kpre  = (fun (i : natlt lena) ->
      Cell a (idx1 i) |-> (acc1 sa i) ** b |-> Frac (fb /. lena) sb);
    kpost = (fun (i : natlt lena) ->
      Cell a (idx1 i) |-> (f (acc1 sa i) (acc1 sb i)) ** b |-> Frac (fb /. lena) sb);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn map_gpu2
  (#et : Type0)
  (f : et -> et -> et)
  (lena : szp { lena <= max_blocks * max_threads })
  (#la : layout1 lena) {| ctlayout la |}
  (#lb : layout1 lena) {| ctlayout lb |}
  (a : array1 et la)
  (b : array1 et lb)
  (#_ : squash (is_global a))
  (#_ : squash (is_global b))
  (#sa : erased (chest1 et lena))
  (#sb : erased (chest1 et lena))
  (#fb : perm)
  norewrite
  preserves cpu ** on gpu_loc (b |-> Frac fb sb)
  requires  on gpu_loc (a |-> sa)
  ensures   on gpu_loc (a |-> chest1_map2 f sa sb)
{
  launch_sync (kmap2 f lena a b);
}
