module Kuiper.Matrix3
#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix
module A = Kuiper.VArray
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

let gpu_matrix3 (et:Type0) (#d0 #d1 #d2 : nat) (l : mlayout3 d0 d1 d2) : Type0 =
  A.varray (aview3_from_mlayout3 et l)

let is_global_matrix3 (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2) (a : gpu_matrix3 et l) : prop =
  A.is_global_varray a

let from_array (#et : Type0) (#d0 #d1 #d2 : erased nat) (l : mlayout3 d0 d1 d2) (a : gpu_array et (mlayout3_size l)) : gpu_matrix3 et l =
  A.from_array (aview3_from_mlayout3 et l) a

let core (#et : Type0) (#d0 #d1 #d2 : erased nat) (#l : mlayout3 d0 d1 d2) (a : gpu_matrix3 et l) : gpu_array et (mlayout3_size l) =
  A.core a

let lem_core_from_array #et #d0 #d1 #d2 #l a = ()
let lem_from_array_core #et #d0 #d1 #d2 l p = ()

let gpu_matrix3_pts_to
  (#et:Type) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  ([@@@mkey] a : gpu_matrix3 et l)
  (#[T.exact (`1.0R)] f : perm)
  (v : earray3 et d0 d1 d2)
  : slprop
  = A.varray_pts_to a #f v

instance is_send_across_global_matrix3
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l { is_global_matrix3 a })
  (#f : perm) (v : earray3 et d0 d1 d2)
  : is_send_across gpu_of (gpu_matrix3_pts_to a #f v)
  = solve

inline_for_extraction noextract
fn gpu_matrix3_read
  (#et:Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : mlayout3 d0 d1 d2) {| clayout3 l |}
  (a : gpu_matrix3 et l)
  (i : szlt d0) (j : szlt d1) (k : szlt d2)
  (#f : perm)
  (#v : erased (earray3 et d0 d1 d2))
  preserves a |-> Frac f v
  returns r : et
  ensures pure (r == acc3 v i j k)
{
  unfold gpu_matrix3_pts_to a #f v;
  let r = A.varray_read a (i, j, k);
  fold gpu_matrix3_pts_to a #f v;
  r
}

inline_for_extraction noextract
fn gpu_matrix3_write
  (#et:Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : mlayout3 d0 d1 d2) {| clayout3 l |}
  (a : gpu_matrix3 et l)
  (i : szlt d0) (j : szlt d1) (k : szlt d2)
  (r : et)
  (#v : erased (earray3 et d0 d1 d2))
  requires a |-> v
  ensures  a |-> mk3 (fun i' j' k' ->
    if i'=i && j'=j && k'=k then r else acc3 v i' j' k')
{
  unfold gpu_matrix3_pts_to a v;
  A.varray_write a (i, j, k) r;
  fold gpu_matrix3_pts_to a (mk3 (fun i' j' k' ->
    if i'=i && j'=j && k'=k then r else acc3 v i' j' k'));
}

(* ============ CELL-LEVEL ============ *)

let gpu_matrix3_pts_to_cell
  (#et:Type) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  ([@@@mkey] a : gpu_matrix3 et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] i : natlt d0) ([@@@mkey] j : natlt d1) ([@@@mkey] k : natlt d2)
  (v : et)
  : slprop
  = A.varray_pts_to_cell a #f (i, j, k) v

let gpu_matrix3_pts_to_cell_eq
  (#et:Type) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l)
  (i : natlt d0) (j : natlt d1) (k : natlt d2)
  (f : perm) (v : et)
  : Lemma (gpu_matrix3_pts_to_cell a #f i j k v
           ==
           gpu_pts_to_cell (core a) #f (l.map3.f (i, j, k)) v)
  = A.varray_pts_to_cell_eq a (i, j, k) f v

ghost
fn gpu_matrix3_explode
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l)
  (#f : perm) (#v : earray3 et d0 d1 d2)
  requires a |-> Frac f v
  ensures
    forall+ (i : natlt d0) (j : natlt d1) (k : natlt d2).
      gpu_matrix3_pts_to_cell a #f i j k (acc3 v i j k)
{
  unfold gpu_matrix3_pts_to a #f v;
  A.varray_explode a;
  forevery_rw_type _ (natlt d0 & natlt d1 & natlt d2) _;
  ghost
  fn aux (ijk : natlt d0 & natlt d1 & natlt d2)
    requires A.varray_pts_to_cell a #f ijk ((aview3_from_mlayout3 et l).ctn.acc v ijk)
    ensures  gpu_matrix3_pts_to_cell a #f ijk._1 ijk._2 ijk._3 (acc3 v ijk._1 ijk._2 ijk._3)
  {
    rewrite each ijk as (ijk._1, ijk._2, ijk._3);
    fold gpu_matrix3_pts_to_cell a #f ijk._1 ijk._2 ijk._3 (acc3 v ijk._1 ijk._2 ijk._3);
  };
  forevery_map _ _ aux;
  forevery_unflatten3' _;
}

ghost
fn gpu_matrix3_implode
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l)
  (#f : perm) (#v : earray3 et d0 d1 d2)
  requires
    pure (SZ.fits (mlayout3_size l))
  requires
    forall+ (i : natlt d0) (j : natlt d1) (k : natlt d2).
      gpu_matrix3_pts_to_cell a #f i j k (acc3 v i j k)
  ensures
    a |-> Frac f v
{
  forevery_flatten3'
    (fun (ijk : natlt d0 & natlt d1 & natlt d2) ->
      gpu_matrix3_pts_to_cell a #f ijk._1 ijk._2 ijk._3 (acc3 v ijk._1 ijk._2 ijk._3));
  forevery_ext
    (fun (ijk : natlt d0 & natlt d1 & natlt d2) ->
      gpu_matrix3_pts_to_cell a #f ijk._1 ijk._2 ijk._3 (acc3 v ijk._1 ijk._2 ijk._3))
    (fun (ijk : natlt d0 & natlt d1 & natlt d2) ->
      A.varray_pts_to_cell a #f ijk ((aview3_from_mlayout3 et l).ctn.acc v ijk));
  A.varray_implode a;
  fold gpu_matrix3_pts_to a #f v;
}

ghost
fn gpu_matrix3_pts_to_ref
  (#et:Type) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l)
  (#f : perm) (#v : erased (earray3 et d0 d1 d2))
  preserves a |-> Frac f v
  ensures pure (SZ.fits (mlayout3_size l))
{
  unfold gpu_matrix3_pts_to a #f v;
  A.varray_pts_to_ref a;
  fold gpu_matrix3_pts_to a #f v;
}

ghost
fn gpu_matrix3_share_n
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (k : pos)
  (#f : perm) (#v : earray3 et d0 d1 d2)
  requires a |-> Frac f v
  ensures  forall+ (_:natlt k). a |-> Frac (f /. k) v
{
  unfold gpu_matrix3_pts_to a #f v;
  A.varray_share_n a k;
  forevery_map
    (fun (i:natlt k) -> A.varray_pts_to a #(f /. k) v)
    (fun (i:natlt k) -> gpu_matrix3_pts_to a #(f /. k) v)
    fn i { fold gpu_matrix3_pts_to a #(f /. k) v };
}

ghost
fn gpu_matrix3_gather_n
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (k : pos)
  (#f : perm) (#v : earray3 et d0 d1 d2)
  requires forall+ (_:natlt k). a |-> Frac (f /. k) v
  ensures  a |-> Frac f v
{
  forevery_map
    (fun (i:natlt k) -> gpu_matrix3_pts_to a #(f /. k) v)
    (fun (i:natlt k) -> A.varray_pts_to a #(f /. k) v)
    fn i { unfold gpu_matrix3_pts_to a #(f /. k) v };
  A.varray_gather_n a k;
  fold gpu_matrix3_pts_to a #f v;
}

(* ============ SLICING ============ *)

let slice0_matrix
  (#et : Type) (#d0 #d1 #d2 : erased nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (i : enatlt d0)
  : gpu_matrix et (slice0_mlayout l i)
  = Matrix.from_array (slice0_mlayout l i) (core a)

let slice1_matrix
  (#et : Type) (#d0 #d1 #d2 : erased nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (j : enatlt d1)
  : gpu_matrix et (slice1_mlayout l j)
  = Matrix.from_array (slice1_mlayout l j) (core a)

let slice2_matrix
  (#et : Type) (#d0 #d1 #d2 : erased nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (k : enatlt d2)
  : gpu_matrix et (slice2_mlayout l k)
  = Matrix.from_array (slice2_mlayout l k) (core a)

(* Cell equivalence between slice and array3 *)
let slice0_cell_eq
  (#et:Type) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (i : natlt d0)
  (j : natlt d1) (k : natlt d2) (f : perm) (v : et)
  : Lemma (gpu_matrix_pts_to_cell (slice0_matrix a i) #f j k v
           ==
           gpu_matrix3_pts_to_cell a #f i j k v)
  = gpu_matrix_pts_to_cell_eq (slice0_matrix a i) j k f v;
    gpu_matrix3_pts_to_cell_eq a i j k f v

let slice1_cell_eq
  (#et:Type) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (j : natlt d1)
  (i : natlt d0) (k : natlt d2) (f : perm) (v : et)
  : Lemma (gpu_matrix_pts_to_cell (slice1_matrix a j) #f i k v
           ==
           gpu_matrix3_pts_to_cell a #f i j k v)
  = gpu_matrix_pts_to_cell_eq (slice1_matrix a j) i k f v;
    gpu_matrix3_pts_to_cell_eq a i j k f v

let slice2_cell_eq
  (#et:Type) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (k : natlt d2)
  (i : natlt d0) (j : natlt d1) (f : perm) (v : et)
  : Lemma (gpu_matrix_pts_to_cell (slice2_matrix a k) #f i j v
           ==
           gpu_matrix3_pts_to_cell a #f i j k v)
  = gpu_matrix_pts_to_cell_eq (slice2_matrix a k) i j f v;
    gpu_matrix3_pts_to_cell_eq a i j k f v

(* ============ SLICE0 EXTRACTION ============ *)

ghost
fn gpu_matrix3_extract_slice0
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (i : enatlt d0)
  (#v : earray3 et d0 d1 d2) (#f : perm)
  requires a |-> Frac f v
  ensures
    factored
      (slice0_matrix a i |-> Frac f (earray3_slice0 v i))
      (a |-> Frac f v)
{
  gpu_matrix3_pts_to_ref a;
  gpu_matrix3_explode a;

  (* Separate slice i from the rest *)
  ghost
  fn extract_i (i' : natlt d0)
    norewrite
    requires
      forall+ (j : natlt d1) (k : natlt d2).
        gpu_matrix3_pts_to_cell a #f i' j k (acc3 v i' j k)
    ensures
      forall+ (j : natlt d1) (k : natlt d2).
        gpu_matrix3_pts_to_cell a #f i' j k (acc3 v i' j k)
  { () };

  forevery_remove _ (reveal i <: natlt d0);

  (* The removed slice: forall+ j k. cell a i j k ... *)
  (* The rest: forall+ (x : natlt d0 { x =!= i }). forall+ j k. cell ... *)

  (* Convert the removed slice to matrix cells *)
  ghost
  fn to_mat_cell (j : natlt d1) (k : natlt d2)
    requires gpu_matrix3_pts_to_cell a #f (reveal i) j k (acc3 v (reveal i) j k)
    ensures  gpu_matrix_pts_to_cell (slice0_matrix a i) #f j k (macc (earray3_slice0 v i) j k)
  {
    slice0_cell_eq a (reveal i) j k f (acc3 v (reveal i) j k);
    rewrite gpu_matrix3_pts_to_cell a #f (reveal i) j k (acc3 v (reveal i) j k)
         as gpu_matrix_pts_to_cell (slice0_matrix a i) #f j k (macc (earray3_slice0 v i) j k);
  };
  forevery_map_2 _ _ to_mat_cell;

  gpu_matrix_implode (slice0_matrix a i);

  (* Build trade *)
  ghost
  fn restore ()
    norewrite
    requires
      forall+ (x : natlt d0 { ~(eq2 #(natlt d0) x (reveal i)) }).
        forall+ (j : natlt d1) (k : natlt d2).
          gpu_matrix3_pts_to_cell a #f x j k (acc3 v x j k)
    requires
      slice0_matrix a i |-> Frac f (earray3_slice0 v i)
    ensures
      a |-> Frac f v
  {
    gpu_matrix_explode (slice0_matrix a i);

    ghost
    fn from_mat_cell (j : natlt d1) (k : natlt d2)
      requires gpu_matrix_pts_to_cell (slice0_matrix a i) #f j k (macc (earray3_slice0 v i) j k)
      ensures  gpu_matrix3_pts_to_cell a #f (reveal i) j k (acc3 v (reveal i) j k)
    {
      slice0_cell_eq a (reveal i) j k f (acc3 v (reveal i) j k);
      rewrite gpu_matrix_pts_to_cell (slice0_matrix a i) #f j k (macc (earray3_slice0 v i) j k)
           as gpu_matrix3_pts_to_cell a #f (reveal i) j k (acc3 v (reveal i) j k);
    };
    forevery_map_2 _ _ from_mat_cell;

    forevery_insert
      #(natlt d0) #(fun (x : natlt d0) -> ~(eq2 #(natlt d0) x (reveal i)))
      (fun (i' : natlt d0) ->
        forall+ (j : natlt d1) (k : natlt d2).
          gpu_matrix3_pts_to_cell a #f i' j k (acc3 v i' j k))
      (reveal i);
    forevery_unrefine _;

    gpu_matrix3_implode a;
  };

  Pulse.Lib.Trade.intro_trade _ _ _ restore;
}

ghost
fn gpu_matrix3_restore_slice0
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (i : enatlt d0)
  (#v : earray3 et d0 d1 d2) (#f : perm)
  requires factored (slice0_matrix a i |-> Frac f (earray3_slice0 v i)) (a |-> Frac f v)
  ensures a |-> Frac f v
{ unfold factored _ _; ambig_trade_elim () }

(* slice1 and slice2 follow the same pattern *)
ghost
fn gpu_matrix3_extract_slice1
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (j : enatlt d1)
  (#v : earray3 et d0 d1 d2) (#f : perm)
  requires a |-> Frac f v
  ensures
    factored
      (slice1_matrix a j |-> Frac f (earray3_slice1 v j))
      (a |-> Frac f v)
{
  gpu_matrix3_pts_to_ref a;
  gpu_matrix3_explode a;

  (* For each i, remove j from the inner quantifier *)
  ghost
  fn extract_j_inner (i : natlt d0)
    norewrite
    requires
      forall+ (j' : natlt d1) (k : natlt d2).
        gpu_matrix3_pts_to_cell a #f i j' k (acc3 v i j' k)
    ensures
      (forall+ (k : natlt d2).
        gpu_matrix3_pts_to_cell a #f i (reveal j) k (acc3 v i (reveal j) k)) **
      (forall+ (x : natlt d1 { ~(eq2 #(natlt d1) x (reveal j)) }) (k : natlt d2).
        gpu_matrix3_pts_to_cell a #f i x k (acc3 v i x k))
  {
    forevery_remove _ (reveal j <: natlt d1);
  };
  forevery_map _ _ extract_j_inner;
  forevery_unzip _ _;

  ghost
  fn to_mat_cell (i : natlt d0) (k : natlt d2)
    requires gpu_matrix3_pts_to_cell a #f i (reveal j) k (acc3 v i (reveal j) k)
    ensures  gpu_matrix_pts_to_cell (slice1_matrix a j) #f i k (macc (earray3_slice1 v j) i k)
  {
    slice1_cell_eq a (reveal j) i k f (acc3 v i (reveal j) k);
    rewrite gpu_matrix3_pts_to_cell a #f i (reveal j) k (acc3 v i (reveal j) k)
         as gpu_matrix_pts_to_cell (slice1_matrix a j) #f i k (macc (earray3_slice1 v j) i k);
  };
  forevery_map_2 _ _ to_mat_cell;
  gpu_matrix_implode (slice1_matrix a j);

  ghost
  fn restore ()
    norewrite
    requires
      forall+ (i : natlt d0) (x : natlt d1 { ~(eq2 #(natlt d1) x (reveal j)) }) (k : natlt d2).
        gpu_matrix3_pts_to_cell a #f i x k (acc3 v i x k)
    requires
      slice1_matrix a j |-> Frac f (earray3_slice1 v j)
    ensures a |-> Frac f v
  {
    gpu_matrix_explode (slice1_matrix a j);
    ghost
    fn from_mat_cell (i : natlt d0) (k : natlt d2)
      requires gpu_matrix_pts_to_cell (slice1_matrix a j) #f i k (macc (earray3_slice1 v j) i k)
      ensures  gpu_matrix3_pts_to_cell a #f i (reveal j) k (acc3 v i (reveal j) k)
    {
      slice1_cell_eq a (reveal j) i k f (acc3 v i (reveal j) k);
      rewrite gpu_matrix_pts_to_cell (slice1_matrix a j) #f i k (macc (earray3_slice1 v j) i k)
           as gpu_matrix3_pts_to_cell a #f i (reveal j) k (acc3 v i (reveal j) k);
    };
    forevery_map_2 _ _ from_mat_cell;

    forevery_zip
      (fun (i : natlt d0) ->
        forall+ (k : natlt d2).
          gpu_matrix3_pts_to_cell a #f i (reveal j) k (acc3 v i (reveal j) k))
      (fun (i : natlt d0) ->
        forall+ (x : natlt d1 { ~(eq2 #(natlt d1) x (reveal j)) }) (k : natlt d2).
          gpu_matrix3_pts_to_cell a #f i x k (acc3 v i x k));
    ghost
    fn insert_j_inner (i : natlt d0)
      norewrite
      requires
        (forall+ (k : natlt d2).
          gpu_matrix3_pts_to_cell a #f i (reveal j) k (acc3 v i (reveal j) k)) **
        (forall+ (x : natlt d1 { ~(eq2 #(natlt d1) x (reveal j)) }) (k : natlt d2).
          gpu_matrix3_pts_to_cell a #f i x k (acc3 v i x k))
      ensures
        forall+ (j' : natlt d1) (k : natlt d2).
          gpu_matrix3_pts_to_cell a #f i j' k (acc3 v i j' k)
    {
      forevery_insert
        #(natlt d1) #(fun (x : natlt d1) -> ~(eq2 #(natlt d1) x (reveal j)))
        (fun (j' : natlt d1) ->
          forall+ (k : natlt d2). gpu_matrix3_pts_to_cell a #f i j' k (acc3 v i j' k))
        (reveal j);
      forevery_unrefine _;
    };
    forevery_map _ _ insert_j_inner;
    gpu_matrix3_implode a;
  };
  Pulse.Lib.Trade.intro_trade _ _ _ restore;
}

ghost
fn gpu_matrix3_restore_slice1
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (j : enatlt d1)
  (#v : earray3 et d0 d1 d2) (#f : perm)
  requires factored (slice1_matrix a j |-> Frac f (earray3_slice1 v j)) (a |-> Frac f v)
  ensures a |-> Frac f v
{ unfold factored _ _; ambig_trade_elim () }

ghost
fn gpu_matrix3_extract_slice2
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (k : enatlt d2)
  (#v : earray3 et d0 d1 d2) (#f : perm)
  requires a |-> Frac f v
  ensures
    factored
      (slice2_matrix a k |-> Frac f (earray3_slice2 v k))
      (a |-> Frac f v)
{
  gpu_matrix3_pts_to_ref a;
  gpu_matrix3_explode a;

  (* For each (i, j), remove k from the innermost quantifier *)
  ghost
  fn extract_k_inner (i : natlt d0) (j : natlt d1)
    norewrite
    requires
      forall+ (k' : natlt d2).
        gpu_matrix3_pts_to_cell a #f i j k' (acc3 v i j k')
    ensures
      gpu_matrix3_pts_to_cell a #f i j (reveal k) (acc3 v i j (reveal k)) **
      (forall+ (x : natlt d2 { ~(eq2 #(natlt d2) x (reveal k)) }).
        gpu_matrix3_pts_to_cell a #f i j x (acc3 v i j x))
  {
    forevery_remove _ (reveal k <: natlt d2);
  };
  forevery_map_2 _ _ extract_k_inner;
  forevery_unzip_2
    (fun (i : natlt d0) (j : natlt d1) ->
      gpu_matrix3_pts_to_cell a #f i j (reveal k) (acc3 v i j (reveal k)))
    (fun (i : natlt d0) (j : natlt d1) ->
      forall+ (x : natlt d2 { ~(eq2 #(natlt d2) x (reveal k)) }).
        gpu_matrix3_pts_to_cell a #f i j x (acc3 v i j x));

  ghost
  fn to_mat_cell (i : natlt d0) (j : natlt d1)
    requires gpu_matrix3_pts_to_cell a #f i j (reveal k) (acc3 v i j (reveal k))
    ensures  gpu_matrix_pts_to_cell (slice2_matrix a k) #f i j (macc (earray3_slice2 v k) i j)
  {
    slice2_cell_eq a (reveal k) i j f (acc3 v i j (reveal k));
    rewrite gpu_matrix3_pts_to_cell a #f i j (reveal k) (acc3 v i j (reveal k))
         as gpu_matrix_pts_to_cell (slice2_matrix a k) #f i j (macc (earray3_slice2 v k) i j);
  };
  forevery_map_2 _ _ to_mat_cell;
  gpu_matrix_implode (slice2_matrix a k);

  ghost
  fn restore ()
    norewrite
    requires
      forall+ (i : natlt d0) (j : natlt d1) (x : natlt d2 { ~(eq2 #(natlt d2) x (reveal k)) }).
        gpu_matrix3_pts_to_cell a #f i j x (acc3 v i j x)
    requires
      slice2_matrix a k |-> Frac f (earray3_slice2 v k)
    ensures a |-> Frac f v
  {
    gpu_matrix_explode (slice2_matrix a k);
    ghost
    fn from_mat_cell (i : natlt d0) (j : natlt d1)
      requires gpu_matrix_pts_to_cell (slice2_matrix a k) #f i j (macc (earray3_slice2 v k) i j)
      ensures  gpu_matrix3_pts_to_cell a #f i j (reveal k) (acc3 v i j (reveal k))
    {
      slice2_cell_eq a (reveal k) i j f (acc3 v i j (reveal k));
      rewrite gpu_matrix_pts_to_cell (slice2_matrix a k) #f i j (macc (earray3_slice2 v k) i j)
           as gpu_matrix3_pts_to_cell a #f i j (reveal k) (acc3 v i j (reveal k));
    };
    forevery_map_2 _ _ from_mat_cell;

    forevery_zip_2
      (fun (i : natlt d0) (j : natlt d1) ->
        gpu_matrix3_pts_to_cell a #f i j (reveal k) (acc3 v i j (reveal k)))
      (fun (i : natlt d0) (j : natlt d1) ->
        forall+ (x : natlt d2 { ~(eq2 #(natlt d2) x (reveal k)) }).
          gpu_matrix3_pts_to_cell a #f i j x (acc3 v i j x));
    ghost
    fn insert_k_inner (i : natlt d0) (j : natlt d1)
      norewrite
      requires
        gpu_matrix3_pts_to_cell a #f i j (reveal k) (acc3 v i j (reveal k)) **
        (forall+ (x : natlt d2 { ~(eq2 #(natlt d2) x (reveal k)) }).
          gpu_matrix3_pts_to_cell a #f i j x (acc3 v i j x))
      ensures
        forall+ (k' : natlt d2).
          gpu_matrix3_pts_to_cell a #f i j k' (acc3 v i j k')
    {
      forevery_insert
        #(natlt d2) #(fun (x : natlt d2) -> ~(eq2 #(natlt d2) x (reveal k)))
        (fun (k' : natlt d2) -> gpu_matrix3_pts_to_cell a #f i j k' (acc3 v i j k'))
        (reveal k);
      forevery_unrefine _;
    };
    forevery_map_2 _ _ insert_k_inner;
    gpu_matrix3_implode a;
  };
  Pulse.Lib.Trade.intro_trade _ _ _ restore;
}

ghost
fn gpu_matrix3_restore_slice2
  (#et:Type0) (#d0 #d1 #d2 : nat) (#l : mlayout3 d0 d1 d2)
  (a : gpu_matrix3 et l) (k : enatlt d2)
  (#v : earray3 et d0 d1 d2) (#f : perm)
  requires factored (slice2_matrix a k |-> Frac f (earray3_slice2 v k)) (a |-> Frac f v)
  ensures a |-> Frac f v
{ unfold factored _ _; ambig_trade_elim () }
