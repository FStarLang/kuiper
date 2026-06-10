module Kuiper.Tensor
#lang-pulse

open Kuiper
open Kuiper.Index
module A = Kuiper.VArray
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

let tensor (et : Type0) (#r : nat) (#d : idesc r) (l : tlayout d) : Type0 =
  A.varray (tensor_aview et l)

let is_global
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l) : prop
  = A.is_global a

inline_for_extraction noextract
let from_array
  (#et : Type0) (#r : erased nat) (#d : idesc r)
  (l : tlayout d)
  (a : larray et (tlayout_size l))
  : tensor et l
  = A.from_array (tensor_aview et l) a

inline_for_extraction noextract
let core
  (#et : Type0) (#r : erased nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : larray et (tlayout_size l)
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
  (p : larray et (tlayout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let lem_is_global_iff_core
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]
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
  (a : tensor et l { is_global a })
  (#f : perm) (s : chest d et)
  : is_send_across gpu_of (tensor_pts_to a #f s)
  = solve

(* This is slightly odd, the user needs to give the total size
instead of each dimension. *)
inline_for_extraction noextract
fn alloc0
  (#et:Type) {| sized et |}
  (#r : nat) (#d : idesc r)
  (s : szp{SZ.v s == sizeof d})
  (l : tlayout d { is_full l })
  preserves
    cpu
  returns
    p : tensor et l
  ensures
    exists* em. on gpu_loc (p |-> em)
  ensures
    pure (is_global p) **
    pure (is_full_array (core p))
{
  let t = A.varray_alloc0 #et s (tensor_aview et l);
  with em. assert on gpu_loc (A.varray_pts_to t em);
  rewrite on gpu_loc (A.varray_pts_to t em)
       as on gpu_loc (tensor_pts_to t em);
  t
}

inline_for_extraction noextract
fn free
  (#et:Type)
  (#r : nat) (#d : idesc r)
  (#l : tlayout d { is_full l })
  (p : tensor et l)
  (#em : chest d et)
  preserves
    cpu
  requires
    pure (is_full_array (core p)) **
    on gpu_loc (p |-> em)
  ensures emp
{
  rewrite on gpu_loc (tensor_pts_to p em)
       as on gpu_loc (A.varray_pts_to p  em);
  A.varray_free p;
}

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
fn tensor_pts_to_eq
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f1 f2 : perm)
  (#s1 #s2 : chest d et)
  requires
    tensor_pts_to a #f1 s1 **
    tensor_pts_to a #f2 s2
  ensures
    tensor_pts_to a #f1 s2 **
    tensor_pts_to a #f2 s2
{
  unfold tensor_pts_to a #f1 s1;
  unfold tensor_pts_to a #f2 s2;
  A.varray_pts_to_eq a f2;
  fold tensor_pts_to a #f1 s2;
  fold tensor_pts_to a #f2 s2;
}

ghost
fn tensor_concr
  (#et:Type)
  (#r : nat) (#d : idesc r)
  (#l : tlayout d { is_full l })
  (g : tensor et l)
  (#s : chest d et)
  (#f : perm)
  requires
    g |-> Frac f s
  ensures
    core g |-> Frac f (to_seq l s)
{
  unfold tensor_pts_to g #f s;
  A.varray_concr g;
  to_seq_rel l s;
  rewrite A.core g |-> Frac f (A.to_seq (tensor_aview et l) s)
       as core g |-> Frac f (to_seq l s);
}

ghost
fn tensor_abs
  (#et:Type)
  (#r : nat) (#d : idesc r)
  (l : tlayout d { is_full l })
  (p : larray et (tlayout_size l))
  (#f : perm)
  (#s : chest d et)
  requires
    p |-> Frac f (to_seq l s)
  ensures
    from_array l p |-> Frac f s
{
  to_seq_rel l s;
  rewrite
    p |-> Frac f (to_seq l s)
  as
    p |-> Frac f (A.to_seq (tensor_aview et l) s);
  A.varray_abs (tensor_aview et l) p;
  fold tensor_pts_to (from_array l p) #f s;
}

ghost
fn tensor_abs'
  (#et:Type)
  (#r : nat) (#d : idesc r)
  (l : tlayout d { is_full l })
  (p : larray et (tlayout_size l))
  (#f : perm)
  (#s : lseq et (tlayout_size l))
  requires
    p |-> Frac f s
  ensures
    from_array l p |-> Frac f (from_seq l s)
{
  rewrite each s as to_seq l (from_seq l s);
  tensor_abs l p;
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
  : Lemma (Cell a i |-> Frac f v
           ==
           pts_to_cell (core a) #f (l.imap.f i) v)
  = A.varray_pts_to_cell_eq a i f v

instance is_send_across_global_tensor_cell
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l { is_global a })
  (#f : perm) (i : abs d) (v : et)
  : is_send_across gpu_of (tensor_pts_to_cell a #f i v)
  = solve

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
      Cell a i |-> Frac f (acc s i)

{
  unfold tensor_pts_to a #f s;
  A.varray_explode a;

  forevery_rw_type _ (abs d) _;
  forevery_ext
    (fun (i : abs d) ->
      Cell a i |-> Frac f ((tensor_aview et l).ctn.acc s i))
    (fun (i : abs d) ->
      Cell a i |-> Frac f (acc s i));
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
      Cell a i |-> Frac f (acc s i)
  ensures
    a |-> Frac f s
{
  forevery_ext
    (fun (i : abs d) ->
      Cell a i |-> Frac f (acc s i))
    (fun (i : abs d) ->
      Cell a i |-> Frac f ((tensor_aview et l).ctn.acc s i));
  forevery_rw_type _ (tensor_aview et l).iview.ait _;
  A.varray_implode a;
  fold tensor_pts_to a #f s;
}

ghost
fn tensor_ilower
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_size l)) **
    (forall+ (i : abs d).
      pts_to_cell (core a) #f (l.imap.f i) (acc s i))
{
  tensor_pts_to_ref a;
  tensor_explode a;
  forevery_map
    (fun (i : abs d) -> Cell a i |-> Frac f (acc s i))
    (fun (i : abs d) -> pts_to_cell (core a) #f (l.imap.f i) (acc s i))
    fn i {
      tensor_pts_to_cell_eq a i f (acc s i);
      rewrite
        Cell a i |-> Frac f (acc s i)
      as
        pts_to_cell (core a) #f (l.imap.f i) (acc s i);
    };
}

ghost
fn tensor_iraise
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    pure (SZ.fits (tlayout_size l)) **
    (forall+ (i : abs d).
      pts_to_cell (core a) #f (l.imap.f i) (acc s i))
  ensures
    a |-> Frac f s
{
  forevery_map
    (fun (i : abs d) -> pts_to_cell (core a) #f (l.imap.f i) (acc s i))
    (fun (i : abs d) -> Cell a i |-> Frac f (acc s i))
    fn i {
      tensor_pts_to_cell_eq a i f (acc s i);
      rewrite
        pts_to_cell (core a) #f (l.imap.f i) (acc s i)
      as
        Cell a i |-> Frac f (acc s i);
    };
  tensor_implode a;
}

inline_for_extraction noextract
fn tensor_read_cell
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (i : conc d)
  (#f : perm)
  (#s : erased et)
  preserves
    Cell a (up i) |-> Frac f s
  returns
    v : et
  ensures
    pure (v == s)
{
  unfold tensor_pts_to_cell a #f (up i) s;
  let v = A.varray_read_cell a i;
  fold tensor_pts_to_cell a #f (up i) s;
  v
}

inline_for_extraction noextract
fn tensor_write_cell
  (#et : Type0) (#r : nat) (#d : idesc r)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (i : conc d)
  (v : et)
  (#s : erased et)
  requires
    Cell a (up i) |-> s
  ensures
    Cell a (up i) |-> v
{
  unfold tensor_pts_to_cell a (up i) s;
  A.varray_write_cell a i v;
  fold tensor_pts_to_cell a (up i) v;
  ()
}

inline_for_extraction noextract
let ctlayout_slice_cimap
  (#n : erased nat) (d : idesc n) (l : tlayout d)
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
  (#n : erased nat) (#d : idesc n) (l : tlayout d)
  {| c : ctlayout l |}
  (i : szlt n) (j : szlt (d @! i))
  : ctlayout (tlayout_slice l i j) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun idx -> ctlayout_slice_cimap d l i j idx);
  }

inline_for_extraction noextract
let sliceof
  (#et : Type0) (#r : erased nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : tensor et (tlayout_slice l i j)
  = from_array (tlayout_slice l i j) (core a)

let lem_sliceof_core
  (#et : Type0) (#r : erased nat) (#d : idesc r)
  (#l : tlayout d)
  (a : tensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : Lemma (core (sliceof a i j) == core a)
          [SMTPat (sliceof a i j)]
  = ()

let tensor_slice_cell_eq
  (#et : Type0) (#r : nat) (#d : idesc r)
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

ghost
fn tensor_apply_bij
  (#et : Type0)
  (#r1 : nat) (#d1 : idesc r1)
  (#r2 : nat) (#d2 : idesc r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (a : tensor et l)
  (m : Chest.t d1 et)
  requires
    a |-> m
  ensures
    from_array (tlayout_bij f l) (core a) |-> Chest.mk d2 (fun a -> Chest.acc m (a <~| f))
{
  // Kuiper.Enumerable.bijection_implies_equal_cardinal (abs d1) (abs d2) f;
  assume pure (sizeof d1 == sizeof d2);
  assert pure (tlayout_size l == tlayout_size (tlayout_bij f l));
  tensor_concr a;
  tensor_abs' (tlayout_bij f l) (core a);
  assert pure (from_seq (tlayout_bij f l) (to_seq l m) `Chest.equal`
               Chest.mk d2 (fun a -> Chest.acc m (a <~| f)));
  ()
}

let fold_bij (#r: nat {r > 1}) (#d: idesc r): (abs d =~ abs (fold_outer d)) = {
  ff = fold_index;
  gg = unfold_index;
  ff_gg = ez;
  gg_ff = ez;
}

ghost
fn tensor_fold_outer
  (#et : Type0)
  (#r: nat {r > 1}) (#d: idesc r)
  (#l: tlayout d)
  (a : tensor et l)
  (m : Chest.t d et)
  requires
    a |-> m
  ensures
    from_array (tlayout_bij fold_bij l) (core a) |-> fold_chest m
{
  tensor_pts_to_ref a;
  tensor_explode a;
  (* forall+ (i : abs d). Cell a i |-> Frac 1.0R (acc m i) *)

  forevery_iso fold_bij
    (fun (i : abs d) -> Cell a i |-> Frac 1.0R (acc m i));
  (* forall+ (j : abs (fold_outer d)).
        Cell a (fold_bij.gg j) |-> Frac 1.0R (acc m (fold_bij.gg j)) *)

  forevery_map
    (fun (j : abs (fold_outer d)) ->
      Cell a (fold_bij.gg j) |-> Frac 1.0R (acc m (fold_bij.gg j)))
    (fun (j : abs (fold_outer d)) ->
      Cell (from_array (tlayout_bij fold_bij l) (core a)) j
        |-> Frac 1.0R (acc (fold_chest m) j))
    fn j {
      tensor_pts_to_cell_eq a (fold_bij.gg j) 1.0R (acc m (fold_bij.gg j));
      tensor_pts_to_cell_eq (from_array (tlayout_bij fold_bij l) (core a)) j 1.0R
        (acc (fold_chest m) j);
      rewrite
        Cell a (fold_bij.gg j) |-> Frac 1.0R (acc m (fold_bij.gg j))
      as
        Cell (from_array (tlayout_bij fold_bij l) (core a)) j
          |-> Frac 1.0R (acc (fold_chest m) j);
    };

  tensor_implode (from_array (tlayout_bij fold_bij l) (core a));
}

ghost
fn tensor_unfold_outer
  (#et : Type0)
  (#r: nat {r > 1}) (#d: idesc r)
  (#l: tlayout d)
  (a : tensor et (tlayout_bij fold_bij l))
  (m : Chest.t (fold_outer d) et)
  requires
    a |-> m
  ensures
    from_array l (core a) |-> unfold_chest m
{
  tensor_pts_to_ref a;
  tensor_explode a;
  (* forall+ (j : abs (fold_outer d)). Cell a j |-> Frac 1.0R (acc m j) *)

  forevery_iso (bij_sym fold_bij)
    (fun (j : abs (fold_outer d)) -> Cell a j |-> Frac 1.0R (acc m j));
  (* forall+ (i : abs d).
        Cell a (fold_bij.ff i) |-> Frac 1.0R (acc m (fold_bij.ff i)) *)

  forevery_map
    (fun (i : abs d) ->
      Cell a (fold_bij.ff i) |-> Frac 1.0R (acc m (fold_bij.ff i)))
    (fun (i : abs d) ->
      Cell (from_array l (core a)) i
        |-> Frac 1.0R (acc (unfold_chest m) i))
    fn i {
      tensor_pts_to_cell_eq a (fold_bij.ff i) 1.0R (acc m (fold_bij.ff i));
      tensor_pts_to_cell_eq (from_array l (core a)) i 1.0R
        (acc (unfold_chest m) i);
      rewrite
        Cell a (fold_bij.ff i) |-> Frac 1.0R (acc m (fold_bij.ff i))
      as
        Cell (from_array l (core a)) i
          |-> Frac 1.0R (acc (unfold_chest m) i);
    };

  tensor_implode (from_array l (core a));
}