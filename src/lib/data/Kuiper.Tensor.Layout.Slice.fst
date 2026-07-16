module Kuiper.Tensor.Layout.Slice
#lang-pulse

open Kuiper
open Kuiper.Shape
open Kuiper.Chest
open Kuiper.Tensor
open Pulse.Lib.Trade

module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

inline_for_extraction noextract
let ctlayout_slice_cimap
  (#n : erased nat) (d : shape n) (l : tlayout d)
  {| c : ctlayout l |}
  (i : szlt n) (j : szlt (d @! i))
  (idx : conc (modulo_i i d))
  : Tot (x : szlt l.ulen{SZ.v x == tlayout_slice_imap d l i j (up idx)}) =
    [@@inline_let] let idx' = c_bring_forward_gg (SZ.v i) d j idx in
    [@@inline_let] let res = c.cimap idx' in
    calc (==) {
      SZ.v res;
      == {}
      SZ.v (c.cimap ((c_conc_bring_forward_bij i d).cgg (j, idx)));
      == {}
      l.imap.f (up ((c_conc_bring_forward_bij i d).cgg (j, idx)));
      == { bring_forward_commute2 i d j idx }
      l.imap.f ((abs_bring_forward_bij i d).gg (SZ.v j, up idx));
      == {}
      tlayout_slice_imap d l i j (up idx);
    };
    res

inline_for_extraction noextract
instance ctlayout_slice
  (#n : erased nat) (#d : shape n) (l : tlayout d)
  {| ctlayout l |}
  (i : erased nat{i < n}) (j : erased nat{j < (d @! i)})
  {| ix : concrete_sz i |} {| jx : concrete_sz j |}
  (#r' : erased nat) (#d' : shape r')
  (#_ : reveal r' == n-1)
  (#_ : d' == modulo_i i d)
  : ctlayout #r' #d' (tlayout_slice l i j) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun idx ->
      ctlayout_slice_cimap d l (concr' ix) (concr' jx) idx);
  }

inline_for_extraction noextract
let sliceof
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : tensor et (tlayout_slice l i j)
  = from_array (tlayout_slice l i j) (core a)

let lem_sliceof_core
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : Lemma (core (sliceof a i j) == core a)
          [SMTPat (sliceof a i j)]
  = ()

let lem_is_global_iff_sliceof
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  : Lemma (is_global (sliceof a i j) <==> is_global a)
          [SMTPat (sliceof a i j)]
  = ()

let tensor_slice_cell_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (k : abs (modulo_i i d)) (f : perm) (v : et)
  : Lemma (Cell (sliceof a i j) k |-> Frac f v
           ==
           Cell a ((abs_bring_forward_bij i d).gg (j, k)) |-> Frac f v)
           [SMTPat (Cell (sliceof a i j) k |-> Frac f v)]
  = tensor_pts_to_cell_eq (sliceof a i j) k f v;
    tensor_pts_to_cell_eq a ((abs_bring_forward_bij i d).gg (j, k)) f v;
    ()

ghost
fn tensor_extract_slice
  (#et : Type0) (#r : nat) (#d : shape r)
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
    (fun (idx : abs d) -> Cell a idx |-> Frac f (acc s idx));

  forevery_unflatten'
    (fun (jk : natlt (d @! i) & abs (modulo_i i d)) ->
      Cell a ((abs_bring_forward_bij i d).gg jk) |-> Frac f (acc s ((abs_bring_forward_bij i d).gg jk)));

  forevery_extract_if_eqtype j
    (fun (j' : natlt (d @! i)) ->
      forall+ (k : abs (modulo_i i d)).
        Cell a ((abs_bring_forward_bij i d).gg (j', k)) |-> Frac f (acc s ((abs_bring_forward_bij i d).gg (j', k))));

  ghost
  fn to_slice_cell (k : abs (modulo_i i d))
    requires
      Cell a ((abs_bring_forward_bij i d).gg (j, k)) |-> Frac f (acc s ((abs_bring_forward_bij i d).gg (j, k)))
    ensures
      Cell (sliceof a i j) k |-> Frac f (acc (chest_slice i j s) k)
  {
    let jk = (abs_bring_forward_bij i d).gg (j, k);
    tensor_pts_to_cell_eq a jk f (acc s jk);
    rewrite
      Cell a ((abs_bring_forward_bij i d).gg (j, k)) |-> Frac f (acc s jk)
    as
      Cell (sliceof a i j) k |-> Frac f (acc (chest_slice i j s) k);
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
            Cell a ((abs_bring_forward_bij i d).gg (j', k)) |-> Frac f (acc s ((abs_bring_forward_bij i d).gg (j', k)))
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
              Cell a ((abs_bring_forward_bij i d).gg (j', k)) |-> Frac f (acc s ((abs_bring_forward_bij i d).gg (j', k)))
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
              Cell a ((abs_bring_forward_bij i d).gg (j', k)) |-> Frac f (acc s ((abs_bring_forward_bij i d).gg (j', k)))
        )
        ensures (
          if op_Equality #(natlt (d @! i)) j' j
          then emp
          else
            forall+ (k : abs (modulo_i i d)).
              Cell a ((abs_bring_forward_bij i d).gg (j', k)) |-> Frac f (acc new_s ((abs_bring_forward_bij i d).gg (j', k)))
        )
      {
        if (j = j') {
          rewrite each op_Equality #(natlt (d @! i)) j' j as true;
          rewrite emp as
            (if op_Equality #(natlt (d @! i)) j' j
             then emp
             else
               forall+ (k : abs (modulo_i i d)).
                 Cell a ((abs_bring_forward_bij i d).gg (j', k)) |-> Frac f (acc new_s ((abs_bring_forward_bij i d).gg (j', k))));
          ()
        } else {
          rewrite each op_Equality #(natlt (d @! i)) j' j as false;
          forevery_map
            (fun (k : abs (modulo_i i d)) ->
              Cell a ((abs_bring_forward_bij i d).gg (j', k)) |-> Frac f (acc s ((abs_bring_forward_bij i d).gg (j', k))))
            (fun (k : abs (modulo_i i d)) ->
              Cell a ((abs_bring_forward_bij i d).gg (j', k)) |-> Frac f (acc new_s ((abs_bring_forward_bij i d).gg (j', k))))
            fn _ {};

          rewrite
            forall+ (k : abs (modulo_i i d)).
              Cell a ((abs_bring_forward_bij i d).gg (j', k)) |-> Frac f (acc new_s ((abs_bring_forward_bij i d).gg (j', k)))
          as
            (if op_Equality #(natlt (d @! i)) j' j
             then emp
             else
               forall+ (k : abs (modulo_i i d)).
                 Cell a ((abs_bring_forward_bij i d).gg (j', k)) |-> Frac f (acc new_s ((abs_bring_forward_bij i d).gg (j', k))));
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
          Cell (sliceof a i j) k |-> Frac f (acc s' k)
        ensures
          Cell a ((abs_bring_forward_bij i d).gg (j, k)) |-> Frac f (acc new_s ((abs_bring_forward_bij i d).gg (j, k)))
      {
        let jk = (abs_bring_forward_bij i d).gg (j, k);
        tensor_pts_to_cell_eq a jk f (acc s jk);
        rewrite
          Cell (sliceof a i j) k |-> Frac f (acc s' k)
        as
          Cell a ((abs_bring_forward_bij i d).gg (j, k)) |-> Frac f (acc new_s jk);
      };
      forevery_map _ _ from_slice_cell;

      (* Combine the two forall+, implode. *)
      forevery_unextract_if_eqtype j
        (fun (j' : natlt (d @! i)) ->
          forall+ (k : abs (modulo_i i d)).
            Cell a ((abs_bring_forward_bij i d).gg (j', k)) |-> Frac f (acc new_s ((abs_bring_forward_bij i d).gg (j', k))));

      forevery_flatten'
        (fun (jk : natlt (d @! i) & abs (modulo_i i d)) ->
          Cell a ((abs_bring_forward_bij i d).gg jk) |-> Frac f (acc new_s ((abs_bring_forward_bij i d).gg jk)));

      forevery_iso_back (abs_bring_forward_bij i d)
        (fun (idx : abs d) -> Cell a idx |-> Frac f (acc new_s idx));

      tensor_implode a;
    };
    Pulse.Lib.Trade.intro_trade _ _ _ restore;
  };
  Pulse.Lib.Forall.intro_forall _ restore';
}

ghost
fn tensor_extract_slice_ro
  (#et : Type0) (#r : nat) (#d : shape r)
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
  (#et : Type0) (#r : nat) (#d : shape r)
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