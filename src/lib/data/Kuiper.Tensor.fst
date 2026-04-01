module Kuiper.Tensor
#lang-pulse

open Kuiper
open Kuiper.Index
module A = Kuiper.VArray
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

let tensor (et : Type0) (#r : nat) (#d : idesc r) (l : tlayout d) : Type0 =
  A.varray (tensor_aview et l)

let is_global_tensor
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) : prop
  = A.is_global_varray a

inline_for_extraction noextract
let from_array
  (#et : Type0) (#r : nat) (#d : idesc r)
  (l : tlayout d)
  (a : gpu_array et (tlayout_size l))
  : tensor et l
  = A.from_array (tensor_aview et l) a

inline_for_extraction noextract
let core
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : gpu_array et (tlayout_size l)
  = A.core a

let lem_core_from_array
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]
  = ()

let lem_from_array_core
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (p : gpu_array et (tlayout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let lem_is_global_iff_core
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures is_global_tensor a <==> is_global_array (core a))
          [SMTPat (is_global_tensor a)]
  = ()

let tensor_pts_to
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  ([@@@mkey] a : tensor et l)
  (#[T.exact (`1.0R)] f : perm)
  (s : chest d et)
  : slprop
  = A.varray_pts_to a #f s

instance is_send_across_global_tensor
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l { is_global_tensor a })
  (#f : perm) (s : chest d et)
  : is_send_across gpu_of (tensor_pts_to a #f s)
  = solve

ghost
fn tensor_pts_to_ref
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm) (#s : chest d et)
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_size l))
{
  unfold tensor_pts_to a #f s;
  A.varray_pts_to_ref a;
  fold tensor_pts_to a #f s;
}

ghost
fn tensor_share_n
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) (k : pos)
  (#f : perm) (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s
{
  unfold tensor_pts_to a #f s;
  A.varray_share_n a k;
  forevery_map
    (fun (i:natlt k) -> A.varray_pts_to a #(f /. k) s)
    (fun (i:natlt k) -> tensor_pts_to a #(f /. k) s)
    fn i { fold tensor_pts_to a #(f /. k) s };
}

ghost
fn tensor_gather_n
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) (k : pos)
  (#f : perm) (#s : chest d et)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s
{
  forevery_map
    (fun (i:natlt k) -> tensor_pts_to a #(f /. k) s)
    (fun (i:natlt k) -> A.varray_pts_to a #(f /. k) s)
    fn i { unfold tensor_pts_to a #(f /. k) s };
  A.varray_gather_n a k;
  fold tensor_pts_to a #f s;
}

inline_for_extraction noextract
fn tensor_read
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (i : conc d)
  (#f : perm)
  (#s : chest d et)
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == acc s (up i))
{
  unfold tensor_pts_to a #f s;
  let v = A.varray_read a i;
  fold tensor_pts_to a #f s;
  v
}

inline_for_extraction noextract
fn tensor_write
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (i : conc d)
  (v : et)
  (#s : chest d et)
  requires
    a |-> s
  ensures
    a |-> upd s (up i) v
{
  unfold tensor_pts_to a s;
  A.varray_write a i v;
  fold tensor_pts_to a;
}

let tensor_pts_to_cell
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  ([@@@mkey] a : tensor et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] i : abs d)
  (v : et)
  : slprop
  = A.varray_pts_to_cell a #f i v

let tensor_pts_to_cell_eq
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) (i : abs d) (f : perm) (v : et)
  : Lemma (tensor_pts_to_cell a #f i v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f i) v)
  = A.varray_pts_to_cell_eq a i f v

ghost
fn tensor_explode
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    forall+ (i : abs d).
      tensor_pts_to_cell a #f i (acc s i)

{
  unfold tensor_pts_to a #f s;
  A.varray_explode a;

  forevery_rw_type _ (abs d) _;
  forevery_ext
    (fun (i : abs d) ->
      A.varray_pts_to_cell a #f i ((tensor_aview et l).ctn.acc s i))
    (fun (i : abs d) ->
      tensor_pts_to_cell a #f i (acc s i));
  ();
}

ghost
fn tensor_implode
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    pure (SZ.fits (tlayout_size l))
  requires
    forall+ (i : abs d).
      tensor_pts_to_cell a #f i (acc s i)
  ensures
    a |-> Frac f s
{
  forevery_ext
    (fun (i : abs d) ->
      tensor_pts_to_cell a #f i (acc s i))
    (fun (i : abs d) ->
      A.varray_pts_to_cell a #f i ((tensor_aview et l).ctn.acc s i));
  forevery_rw_type _ (tensor_aview et l).iview.ait _;
  A.varray_implode a;
  fold tensor_pts_to a #f s;
}

inline_for_extraction noextract
let sliceof
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  : tensor et (tlayout_slice d l i j)
  = from_array (tlayout_slice d l i j) (core a)

let tensor_slice_cell_eq
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (k : abs (modulo_i i d)) (f : perm) (v : et)
  : Lemma (tensor_pts_to_cell (sliceof a i j) #f k v
           ==
           tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j, k)) v)
  = tensor_pts_to_cell_eq (sliceof a i j) k f v;
    tensor_pts_to_cell_eq a ((abs_bring_forward_bij i d).gg (j, k)) f v;
    ()

ghost
fn tensor_extract_slice
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (#f : perm) (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    sliceof a i j |-> Frac f (chest_slice i j s) **
    (forall* (s' : chest (modulo_i i d) et).
      sliceof a i j |-> Frac f s' @==>
      a |-> Frac f (chest_update_slice i j s s'))
{
  (* This proof is terrible. It may be easier by filtering the type of the
  forall+ with a refinement instead of using an if within it. *)
  tensor_pts_to_ref a;
  tensor_explode a;

  forevery_iso (abs_bring_forward_bij i d)
    (fun (idx : abs d) -> tensor_pts_to_cell a #f idx (acc s idx));

  forevery_unflatten'
    (fun (jk : natlt (d @! i) & abs (modulo_i i d)) ->
      tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg jk) (acc s ((abs_bring_forward_bij i d).gg jk)));

  forevery_extract_if_eqtype j
    (fun (j' : natlt (d @! i)) ->
      forall+ (k : abs (modulo_i i d)).
        tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j', k)) (acc s ((abs_bring_forward_bij i d).gg (j', k))));

  ghost
  fn to_slice_cell (k : abs (modulo_i i d))
    requires tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j, k)) (acc s ((abs_bring_forward_bij i d).gg (j, k)))
    ensures  tensor_pts_to_cell (sliceof a i j) #f k (acc (chest_slice i j s) k)
  {
    tensor_pts_to_cell_eq a ((abs_bring_forward_bij i d).gg (j, k)) f (acc s ((abs_bring_forward_bij i d).gg (j, k)));
    rewrite tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j, k)) (acc s ((abs_bring_forward_bij i d).gg (j, k)))
         as gpu_pts_to_cell (core a) #f (l.imap.f ((abs_bring_forward_bij i d).gg (j, k))) (acc s ((abs_bring_forward_bij i d).gg (j, k)));
    tensor_pts_to_cell_eq (sliceof a i j) k f (acc (chest_slice i j s) k);
    rewrite gpu_pts_to_cell (core a) #f (l.imap.f ((abs_bring_forward_bij i d).gg (j, k))) (acc s ((abs_bring_forward_bij i d).gg (j, k)))
         as tensor_pts_to_cell (sliceof a i j) #f k (acc (chest_slice i j s) k);
  };
  forevery_map _ _ to_slice_cell;

  tensor_implode (sliceof a i j);

  ghost
  fn restore' (s' : chest (modulo_i i d) et)
    requires
      forall+ (j' : natlt (d @! i)). (
        if op_Equality #(natlt (d @! i)) j' j
        then emp
        else
          forall+ (k : abs (modulo_i i d)).
            tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j', k)) (acc s ((abs_bring_forward_bij i d).gg (j', k)))
      )
     ensures
       sliceof a i j |-> Frac f s' @==> a |-> Frac f (chest_update_slice i j s s')
  {
    ghost
    fn restore ()
      norewrite
      requires
        forall+ (j' : natlt (d @! i)). (
          if op_Equality #(natlt (d @! i)) j' j
          then emp
          else
            forall+ (k : abs (modulo_i i d)).
              tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j', k)) (acc s ((abs_bring_forward_bij i d).gg (j', k)))
        )
      requires
        sliceof a i j |-> Frac f s'
      ensures
        a |-> Frac f (chest_update_slice i j s s')
    {
      let new_s = chest_update_slice i j s s';
      (* Rewrite the "frame" to make them point to the updated chest
      chest_update_slice i j s s' instead of s. All of them are outside the
      actual modified slice, so this is trivial. *)
      ghost fn rw1 (j' : natlt (d @! i))
        requires (
          if op_Equality #(natlt (d @! i)) j' j
          then emp
          else
            forall+ (k : abs (modulo_i i d)).
              tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j', k)) (acc s ((abs_bring_forward_bij i d).gg (j', k)))
        )
        ensures (
          if op_Equality #(natlt (d @! i)) j' j
          then emp
          else
            forall+ (k : abs (modulo_i i d)).
              tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j', k)) (acc new_s ((abs_bring_forward_bij i d).gg (j', k)))
        )
      {
        if (j = j') {
          rewrite each op_Equality #(natlt (d @! i)) j' j as true;
          rewrite emp as
            (if op_Equality #(natlt (d @! i)) j' j
             then emp
             else
               forall+ (k : abs (modulo_i i d)).
                 tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j', k)) (acc new_s ((abs_bring_forward_bij i d).gg (j', k))));
          ()
        } else {
          rewrite each op_Equality #(natlt (d @! i)) j' j as false;
          forevery_map
            (fun (k : abs (modulo_i i d)) ->
              tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j', k)) (acc s ((abs_bring_forward_bij i d).gg (j', k))))
            (fun (k : abs (modulo_i i d)) ->
              tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j', k)) (acc new_s ((abs_bring_forward_bij i d).gg (j', k))))
            fn _ {};

          rewrite
            forall+ (k : abs (modulo_i i d)).
              tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j', k)) (acc new_s ((abs_bring_forward_bij i d).gg (j', k)))
          as
            (if op_Equality #(natlt (d @! i)) j' j
             then emp
             else
               forall+ (k : abs (modulo_i i d)).
                 tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j', k)) (acc new_s ((abs_bring_forward_bij i d).gg (j', k))));
          ();
        }
      };
      forevery_map _ _ rw1;

      (* Now, bring the cells of sliceof a i j back to cells of a, and rewrite
      them to point to a part of new_s. *)

      tensor_explode (sliceof a i j);
      ghost
      fn from_slice_cell (k : abs (modulo_i i d))
        requires
          tensor_pts_to_cell (sliceof a i j) #f k (acc s' k)
        ensures
          tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j, k)) (acc new_s ((abs_bring_forward_bij i d).gg (j, k)))
      {
        tensor_pts_to_cell_eq (sliceof a i j) k f (acc s' k);
        rewrite tensor_pts_to_cell (sliceof a i j) #f k (acc s' k)
            as gpu_pts_to_cell (core a) #f (l.imap.f ((abs_bring_forward_bij i d).gg (j, k))) (acc new_s ((abs_bring_forward_bij i d).gg (j, k)));
        tensor_pts_to_cell_eq a ((abs_bring_forward_bij i d).gg (j, k)) f (acc new_s ((abs_bring_forward_bij i d).gg (j, k)));
        rewrite gpu_pts_to_cell (core a) #f (l.imap.f ((abs_bring_forward_bij i d).gg (j, k))) (acc new_s ((abs_bring_forward_bij i d).gg (j, k)))
            as tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j, k)) (acc new_s ((abs_bring_forward_bij i d).gg (j, k)));
      };
      forevery_map _ _ from_slice_cell;

      (* Combine the two forall+, implode. *)

      forevery_unextract_if_eqtype j
        (fun (j' : natlt (d @! i)) ->
          forall+ (k : abs (modulo_i i d)).
            tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg (j', k)) (acc new_s ((abs_bring_forward_bij i d).gg (j', k))));

      forevery_flatten'
        (fun (jk : natlt (d @! i) & abs (modulo_i i d)) ->
          tensor_pts_to_cell a #f ((abs_bring_forward_bij i d).gg jk) (acc new_s ((abs_bring_forward_bij i d).gg jk)));

      forevery_iso_back (abs_bring_forward_bij i d)
        (fun (idx : abs d) -> tensor_pts_to_cell a #f idx (acc new_s idx));

      tensor_implode a;
    };
    Pulse.Lib.Trade.intro_trade _ _ _ restore;
  };
  Pulse.Lib.Forall.intro_forall _ restore';
}

ghost
fn tensor_extract_slice_ro
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (#f : perm) (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    factored
      (sliceof a i j |-> Frac f (chest_slice i j s))
      (a |-> Frac f s)
{
  (* Use the RW version, and immediately eliminate the forall*. *)
  tensor_extract_slice a i j;
  Pulse.Lib.Forall.elim_forall (chest_slice i j s);
  assert pure (Kuiper.Chest.equal (chest_update_slice i j s (chest_slice i j s)) s);
  rewrite each chest_update_slice i j s (chest_slice i j s) as s;
}

ghost
fn tensor_restore_slice
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (#f : perm) (#s : chest d et)
  requires
    factored
      (sliceof a i j |-> Frac f (chest_slice i j s))
      (a |-> Frac f s)
  ensures
    a |-> Frac f s
{
  unfold factored _ _;
  ambig_trade_elim ();
}
