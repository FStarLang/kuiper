module Kuiper.Matrix3
#lang-pulse

(* Three-dimensional arrays with a layout, analogous to gpu_matrix.
   Fixing any one index yields an mlayout for the remaining two dimensions,
   allowing slicing into a gpu_matrix. *)

open Kuiper
open Kuiper.Injection
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix
module A = Kuiper.VArray
module V = Kuiper.View
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2
open FStar.Tactics.Typeclasses

(* ============ LAYOUT ============ *)

[@@erasable]
noeq
type mlayout3 (d0 d1 d2 : nat) = {
  len3 : nat;
  map3 : (natlt d0 & natlt d1 & natlt d2 @~> natlt len3);
}

let mlayout3_size (#d0 #d1 #d2 : nat) (l : mlayout3 d0 d1 d2) : GTot nat = l.len3

inline_for_extraction
class clayout3 (#d0 #d1 #d2 : erased nat) (l : mlayout3 d0 d1 d2) = {
  [@@@no_method] c3_len : (x:SZ.t { SZ.v x == l.len3 });
  [@@@no_method] c3_d0  : (x:SZ.t { SZ.v x == reveal d0 });
  [@@@no_method] c3_d1  : (x:SZ.t { SZ.v x == reveal d1 });
  [@@@no_method] c3_d2  : (x:SZ.t { SZ.v x == reveal d2 });
  [@@@no_method] c3_to  : (i:SZ.t{i < d0}) -> (j:SZ.t{j < d1}) -> (k:SZ.t{k < d2})
                           -> r:SZ.t{SZ.v r == l.map3.f (SZ.v i, SZ.v j, SZ.v k)};
}

(* ============ SPEC ============ *)

[@@erasable]
noeq
type earray3 (et:Type) (d0 d1 d2 : nat) =
  | A3 : f:(natlt d0 & natlt d1 & natlt d2 ^->> et)
       -> earray3 et d0 d1 d2

let mk3 (#et:Type) (#d0 #d1 #d2 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> GTot et)
  : earray3 et d0 d1 d2
  = A3 <| FStar.FunctionalExtensionality.on_g _ <| fun (i, j, k) -> f i j k

let acc3 (#et:Type) (#d0 #d1 #d2 : nat)
  (a : earray3 et d0 d1 d2) (i : natlt d0) (j : natlt d1) (k : natlt d2)
  : GTot et
  = a.f (i, j, k)

(* ============ VIEW ============ *)

instance earray3_is_container
  (et:Type) (#d0 #d1 #d2 : nat)
  : Kuiper.Container.container (earray3 et d0 d1 d2) (natlt d0 & natlt d1 & natlt d2) et
  = {
    acc = (fun a (i,j,k) -> acc3 a i j k);
    upd = (fun a (i,j,k) v -> mk3 fun i' j' k' ->
      if i'=i && j'=j && k'=k then v else acc3 a i' j' k');
    l1 = ez;
    l2 = ez;
    ext = (fun c1 c2 _ ->
      assert (FStar.FunctionalExtensionality.feq_g c1.f c2.f));
    from_fun = (fun f -> mk3 fun i j k -> f (i,j,k));
    from_fun_ok = ez;
  }

let aview3_from_mlayout3
  (et : Type) (#d0 #d1 #d2 : nat)
  (l : mlayout3 d0 d1 d2)
  : V.aview et (earray3 et d0 d1 d2)
  = {
    iview = {
      len = l.len3;
      ait = natlt d0 & natlt d1 & natlt d2;
      step = { imap = l.map3; };
    };
    ctn = earray3_is_container et;
  }

(* ============ TYPE ============ *)

inline_for_extraction noextract
val gpu_array3 (et:Type0) (#d0 #d1 #d2 : nat) (l : mlayout3 d0 d1 d2) : Type0

val is_global_array3 (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2) (a : gpu_array3 et l) : prop

inline_for_extraction noextract
val from_array3
  (#et : Type0) (#d0 #d1 #d2 : erased nat)
  (l : mlayout3 d0 d1 d2)
  (a : gpu_array et (mlayout3_size l))
  : gpu_array3 et l

inline_for_extraction noextract
val core3
  (#et : Type0) (#d0 #d1 #d2 : erased nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l)
  : gpu_array et (mlayout3_size l)

val lem_core3_from_array3
  (#et : Type) (#d0 #d1 #d2 : erased nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l)
  : Lemma (ensures from_array3 l (core3 a) == a
                   /\ (is_global_array (core3 a) <==> is_global_array3 a))
          [SMTPat (core3 a)]

val lem_from_array3_core3
  (#et : Type) (#d0 #d1 #d2 : erased nat)
  (l : mlayout3 d0 d1 d2)
  (p : gpu_array et (mlayout3_size l))
  : Lemma (ensures core3 (from_array3 l p) == p
                   /\ (is_global_array3 (from_array3 l p) <==> is_global_array p))
          [SMTPat (from_array3 l p)]

val gpu_array3_pts_to
  (#et:Type) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  ([@@@mkey] a : gpu_array3 et l)
  (#[T.exact (`1.0R)] f : perm)
  (v : earray3 et d0 d1 d2)
  : slprop

instance
val is_send_across_global_array3
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l { is_global_array3 a })
  (#f : perm) (v : earray3 et d0 d1 d2)
  : is_send_across gpu_of (gpu_array3_pts_to a #f v)

unfold
instance has_pts_to_array3 (et : Type) (d0 d1 d2 : erased nat) (l : _)
  : has_pts_to (gpu_array3 et l) (earray3 et d0 d1 d2) = {
  pts_to = gpu_array3_pts_to;
}

(* ============ CONCRETE VIEW INSTANCE ============ *)

inline_for_extraction noextract
let clayout3_imap
  (#d0 #d1 #d2 : erased nat)
  (#l : mlayout3 d0 d1 d2)
  (c : clayout3 l)
  : szlt d0 & szlt d1 & szlt d2 -> szlt l.len3
  = fun (i, j, k) -> c.c3_to i j k

inline_for_extraction noextract
instance cview3_from_clayout3
  (et : Type)
  (#d0 #d1 #d2 : erased nat)
  (l : mlayout3 d0 d1 d2)
  (c : clayout3 l)
  : Kuiper.IView.ciview (aview3_from_mlayout3 et l).iview
  = {
    clen = c.c3_len;
    sch = {
      cit = szlt d0 & szlt d1 & szlt d2;
      bij = Kuiper.Bijection.natural;
    };
    step = {
      cimap = mk_cinj (clayout3_imap c);
      compat = ez;
    };
  }

(* ============ READ / WRITE ============ *)

inline_for_extraction noextract
fn gpu_array3_read
  (#et:Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : mlayout3 d0 d1 d2) {| clayout3 l |}
  (a : gpu_array3 et l)
  (i : szlt d0) (j : szlt d1) (k : szlt d2)
  (#f : perm)
  (#v : erased (earray3 et d0 d1 d2))
  preserves a |-> Frac f v
  returns r : et
  ensures pure (r == acc3 v i j k)

inline_for_extraction noextract
fn gpu_array3_write
  (#et:Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : mlayout3 d0 d1 d2) {| clayout3 l |}
  (a : gpu_array3 et l)
  (i : szlt d0) (j : szlt d1) (k : szlt d2)
  (r : et)
  (#v : erased (earray3 et d0 d1 d2))
  requires a |-> v
  ensures  a |-> mk3 (fun i' j' k' ->
    if i'=i && j'=j && k'=k then r else acc3 v i' j' k')

(* ============ CELL-LEVEL OPERATIONS ============ *)

val gpu_array3_pts_to_cell
  (#et:Type) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  ([@@@mkey] a : gpu_array3 et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] i : natlt d0) ([@@@mkey] j : natlt d1) ([@@@mkey] k : natlt d2)
  (v : et)
  : slprop

val gpu_array3_pts_to_cell_eq
  (#et:Type) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l)
  (i : natlt d0) (j : natlt d1) (k : natlt d2)
  (f : perm) (v : et)
  : Lemma (gpu_array3_pts_to_cell a #f i j k v
           ==
           gpu_pts_to_cell (core3 a) #f (l.map3.f (i, j, k)) v)

ghost
fn gpu_array3_explode
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l)
  (#f : perm) (#v : earray3 et d0 d1 d2)
  requires a |-> Frac f v
  ensures
    forall+ (i : natlt d0) (j : natlt d1) (k : natlt d2).
      gpu_array3_pts_to_cell a #f i j k (acc3 v i j k)

ghost
fn gpu_array3_implode
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l)
  (#f : perm) (#v : earray3 et d0 d1 d2)
  requires
    pure (SZ.fits (mlayout3_size l))
  requires
    forall+ (i : natlt d0) (j : natlt d1) (k : natlt d2).
      gpu_array3_pts_to_cell a #f i j k (acc3 v i j k)
  ensures
    a |-> Frac f v

ghost
fn gpu_array3_pts_to_ref
  (#et:Type) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l)
  (#f : perm) (#v : erased (earray3 et d0 d1 d2))
  preserves a |-> Frac f v
  ensures pure (SZ.fits (mlayout3_size l))

(* ============ SLICING TO MATRIX ============ *)

(* Fix index 0: slice at i, get a matrix over (d1, d2). *)
let slice0_mlayout (#d0 #d1 #d2 : nat) (l : mlayout3 d0 d1 d2) (i : natlt d0)
  : mlayout d1 d2
  = { len = l.len3;
      map = { f = (fun (j, k) -> l.map3.f (i, j, k));
              is_inj = (fun (j1,k1) (j2,k2) -> l.map3.is_inj (i,j1,k1) (i,j2,k2)); }; }

let earray3_slice0 (#et:Type) (#d0 #d1 #d2 : nat)
  (a : earray3 et d0 d1 d2) (i : natlt d0)
  : ematrix et d1 d2
  = mkM fun j k -> acc3 a i j k

val slice0_matrix
  (#et : Type) (#d0 #d1 #d2 : erased nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l) (i : enatlt d0)
  : gpu_matrix et (slice0_mlayout l i)

(* Fix index 1: slice at j, get a matrix over (d0, d2). *)
let slice1_mlayout (#d0 #d1 #d2 : nat) (l : mlayout3 d0 d1 d2) (j : natlt d1)
  : mlayout d0 d2
  = { len = l.len3;
      map = { f = (fun (i, k) -> l.map3.f (i, j, k));
              is_inj = (fun (i1,k1) (i2,k2) -> l.map3.is_inj (i1,j,k1) (i2,j,k2)); }; }

let earray3_slice1 (#et:Type) (#d0 #d1 #d2 : nat)
  (a : earray3 et d0 d1 d2) (j : natlt d1)
  : ematrix et d0 d2
  = mkM fun i k -> acc3 a i j k

val slice1_matrix
  (#et : Type) (#d0 #d1 #d2 : erased nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l) (j : enatlt d1)
  : gpu_matrix et (slice1_mlayout l j)

(* Fix index 2: slice at k, get a matrix over (d0, d1). *)
let slice2_mlayout (#d0 #d1 #d2 : nat) (l : mlayout3 d0 d1 d2) (k : natlt d2)
  : mlayout d0 d1
  = { len = l.len3;
      map = { f = (fun (i, j) -> l.map3.f (i, j, k));
              is_inj = (fun (i1,j1) (i2,j2) -> l.map3.is_inj (i1,j1,k) (i2,j2,k)); }; }

let earray3_slice2 (#et:Type) (#d0 #d1 #d2 : nat)
  (a : earray3 et d0 d1 d2) (k : natlt d2)
  : ematrix et d0 d1
  = mkM fun i j -> acc3 a i j k

val slice2_matrix
  (#et : Type) (#d0 #d1 #d2 : erased nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l) (k : enatlt d2)
  : gpu_matrix et (slice2_mlayout l k)

(* ============ SLICE EXTRACTION WITH OWNERSHIP ============ *)

open Pulse.Lib.Trade

ghost
fn gpu_array3_extract_slice0
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l) (i : enatlt d0)
  (#v : earray3 et d0 d1 d2) (#f : perm)
  requires a |-> Frac f v
  ensures
    factored
      (slice0_matrix a i |-> Frac f (earray3_slice0 v i))
      (a |-> Frac f v)

ghost
fn gpu_array3_restore_slice0
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l) (i : enatlt d0)
  (#v : earray3 et d0 d1 d2) (#f : perm)
  requires
    factored
      (slice0_matrix a i |-> Frac f (earray3_slice0 v i))
      (a |-> Frac f v)
  ensures a |-> Frac f v

ghost
fn gpu_array3_extract_slice1
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l) (j : enatlt d1)
  (#v : earray3 et d0 d1 d2) (#f : perm)
  requires a |-> Frac f v
  ensures
    factored
      (slice1_matrix a j |-> Frac f (earray3_slice1 v j))
      (a |-> Frac f v)

ghost
fn gpu_array3_restore_slice1
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l) (j : enatlt d1)
  (#v : earray3 et d0 d1 d2) (#f : perm)
  requires
    factored
      (slice1_matrix a j |-> Frac f (earray3_slice1 v j))
      (a |-> Frac f v)
  ensures a |-> Frac f v

ghost
fn gpu_array3_extract_slice2
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l) (k : enatlt d2)
  (#v : earray3 et d0 d1 d2) (#f : perm)
  requires a |-> Frac f v
  ensures
    factored
      (slice2_matrix a k |-> Frac f (earray3_slice2 v k))
      (a |-> Frac f v)

ghost
fn gpu_array3_restore_slice2
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_array3 et l) (k : enatlt d2)
  (#v : earray3 et d0 d1 d2) (#f : perm)
  requires
    factored
      (slice2_matrix a k |-> Frac f (earray3_slice2 v k))
      (a |-> Frac f v)
  ensures a |-> Frac f v
