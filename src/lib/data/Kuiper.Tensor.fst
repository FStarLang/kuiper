module Kuiper.Tensor
#lang-pulse

open Kuiper
open Kuiper.Shape
open Kuiper.Bijection
module A = Kuiper.VArray
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

let tensor (et : Type0) (#r : nat) (#d : shape r) (l : tlayout d) : Type0 =
  A.varray (tensor_aview et l)

let is_global
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l) : prop
  = A.is_global a

inline_for_extraction noextract
let from_array
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (l : tlayout d)
  (a : larray et (tlayout_ulen l))
  : tensor et l
  = A.from_array (tensor_aview et l) a

inline_for_extraction noextract
let core
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  : larray et (tlayout_ulen l)
  = A.core a

let lem_core_from_array
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]
  = ()

let lem_from_array_core
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (p : larray et (tlayout_ulen l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let lem_is_global_iff_core
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]
  = ()

let tensor_pts_to
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  ([@@@mkey] a : tensor et l)
  (#[T.exact (`1.0R)] f : perm)
  (s : chest d et)
  : slprop
  = A.varray_pts_to a #f s

instance is_send_across_global_tensor
  (#et : Type0) (#r : nat) (#d : shape r)
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
  (#r : nat) (#d : shape r)
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
  (#r : nat) (#d : shape r)
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
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm) (#s : chest d et)
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_ulen l))
{
  unfold tensor_pts_to a #f s;
  A.varray_pts_to_ref a;
  fold tensor_pts_to a #f s;
}

ghost
fn tensor_pts_to_ref_located
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (#loc : loc_id)
  (#f : perm) (#s : chest d et)
  preserves
    on loc (a |-> Frac f s)
  ensures
    pure (SZ.fits (tlayout_ulen l))
{
  map_loc loc
    #(a |-> Frac f s)
    #(a |-> Frac f s ** pure (SZ.fits (tlayout_ulen l)))
  fn _ {
    tensor_pts_to_ref a;
  };
}

ghost
fn tensor_pts_to_eq
  (#et : Type0) (#r : nat) (#d : shape r)
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
  (#r : nat) (#d : shape r)
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
  (#r : nat) (#d : shape r)
  (l : tlayout d { is_full l })
  (p : larray et (tlayout_ulen l))
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
  (#r : nat) (#d : shape r)
  (l : tlayout d { is_full l })
  (p : larray et (tlayout_ulen l))
  (#f : perm)
  (#s : lseq et (tlayout_ulen l))
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
  (#et : Type0) (#r : nat) (#d : shape r)
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
  (#et : Type0) (#r : nat) (#d : shape r)
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

ghost
fn tensor_gather_n_underspec
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l) (k : pos)
  (#f : perm)
  requires
    forall+ (_:natlt k).
      exists* (s : chest d et). tensor_pts_to a #(f /. k) s
  ensures
    exists* (s : chest d et). tensor_pts_to a #f s
{
  forevery_natlt_pop k _;
  with s. assert tensor_pts_to a #(f /. k) s;
  ghost
  fn aux (_ : natlt (k-1))
    norewrite
    requires
      tensor_pts_to a #(f /. k) s ** (exists* v. tensor_pts_to a #(f /. k) v)
    ensures
      tensor_pts_to a #(f /. k) s ** tensor_pts_to a #(f /. k) s
  {
    tensor_pts_to_eq a (f /. k) #_ #s;
  };
  forevery_map_extra #(natlt (k-1)) (tensor_pts_to a #(f /. k) s)
    (fun (_ : natlt (k-1)) -> exists* v. tensor_pts_to a #(f /. k) v)
    (fun (_ : natlt (k-1)) -> tensor_pts_to a #(f /. k) s)
    aux;
  forevery_natlt_push k _;
  tensor_gather_n a k;
}

inline_for_extraction noextract
fn tensor_read
  (#et : Type0) (#r : erased nat) (#d : shape r)
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
  (#et : Type0) (#r : erased nat) (#d : shape r)
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
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  ([@@@mkey] a : tensor et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] i : abs d)
  (v : et)
  : slprop
  = A.varray_pts_to_cell a #f i v

let tensor_pts_to_cell_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l) (i : abs d) (f : perm) (v : et)
  : Lemma (Cell a i |-> Frac f v
           ==
           pts_to_cell (core a) #f (l.imap.f i) v)
  = A.varray_pts_to_cell_eq a i f v

instance is_send_across_global_tensor_cell
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l { is_global a })
  (#f : perm) (i : abs d) (v : et)
  : is_send_across gpu_of (tensor_pts_to_cell a #f i v)
  = solve

ghost
fn tensor_explode
  (#et : Type0) (#r : nat) (#d : shape r)
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
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    pure (SZ.fits (tlayout_ulen l))
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
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_ulen l)) **
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
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
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
  (#et : Type0) (#r : erased nat) (#d : shape r)
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
  (#et : Type0) (#r : erased nat) (#d : shape r)
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

inline_for_extraction noextract
let ctlayout_bij_cimap
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (fconc: conc d2 -> conc d1)
  (fconc_correct: (x: conc d2) -> up (fconc x) == f.gg (up x))
  (l : tlayout d1) {| c: ctlayout l |}
  (idx: conc d2)
  : Tot (x : szlt l.ulen{SZ.v x == l.imap.f ((f.gg) (up idx))})  =
  fconc_correct idx;
  c.cimap (fconc idx)

inline_for_extraction noextract
instance ctlayout_bij
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2 { all_fit d2 })
  (f : abs d1 =~ abs d2)
  (fconc: conc d2 -> conc d1)
  (fconc_correct: (x: conc d2) -> up (fconc x) == f.gg (up x))
  (l : tlayout d1) {| c: ctlayout l |}
  : ctlayout #r2 #d2 (tlayout_bij f l) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun (idx: conc d2) ->
              fconc_correct idx;
              c.cimap (fconc idx));
  }

ghost
fn tensor_apply_bij
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : tlayout d1) {| is_full l |}
  (a : tensor et l)
  (#fp : perm) (#m : Chest.t d1 et)
  requires
    a |-> Frac fp m
  ensures
    from_array (tlayout_bij f l) (core a) |-> Frac fp (Chest.mk d2 (fun a -> Chest.acc m (a <~| f)))
{
  sizeof_bijection f;
  assert pure (tlayout_size l == tlayout_size (tlayout_bij f l));
  tensor_concr a;
  tensor_abs' (tlayout_bij f l) (core a);
  assert pure (from_seq (tlayout_bij f l) (to_seq l m) `Chest.equal`
               Chest.mk d2 (fun a -> Chest.acc m (a <~| f)));
  ()
}

let fold_bij (#r: nat {r > 1}) (#d: shape r): (abs d =~ abs (fold_outer d)) = {
  ff = fold_index;
  gg = unfold_index;
  ff_gg = ez;
  gg_ff = ez;
}

inline_for_extraction noextract
let unfold_index_conc
  (#r: erased nat {r > 1})
  (#d: shape r { all_fit d }) {| cs: concrete_sz (desc_top2 d)._2 |}
  (i : conc (fold_outer d)): Tot (conc d) =
  let i : szlt (head d * head (tail d)) & conc (tail (tail d)) = i in
  let (ih, it) = i in
  let ih1: szlt (head d) = ih /^ (concr' cs) in
  let ih2: szlt (head (tail d)) = ih %^ (concr' cs) in
  (ih1, (ih2, it))

let all_fit_fold_outer (#r: nat {r > 1}) (#d: shape r { all_fit d }) (#top2_fits: SZ.fits ((desc_top2 d)._1 * (desc_top2 d)._2)):
  Lemma (all_fit (fold_outer d)) = ()

inline_for_extraction noextract
instance ctlayout_fold_outer
  (#r : nat {r > 1}) (#d : shape r { all_fit d })
  (#top2_fits: SZ.fits ((desc_top2 d)._1 * (desc_top2 d)._2))
  (l : tlayout d) {| c: ctlayout l, cs: concrete_sz (desc_top2 d)._2 |}
  : ctlayout #_ #(fold_outer d) (tlayout_fold_outer l) =
  ctlayout_bij (fold_bij #r #d)
    (unfold_index_conc #r #d #cs)
    (fun (x: conc (fold_outer d)) ->
       (() <: squash (up (unfold_index_conc #r #d #cs x) == unfold_index (up x))))
    l

ghost
fn tensor_fold_outer
  (#et : Type0)
  (#r: nat {r > 1}) (#d: shape r)
  (#l: tlayout d)
  (a : tensor et l)
  (#f : perm) (#m : Chest.t d et)
  requires
    a |-> Frac f m
  ensures
    from_array (tlayout_fold_outer l) (core a) |-> Frac f (fold_chest m)
{
  tensor_pts_to_ref a;
  tensor_explode a;
  (* forall+ (i : abs d). Cell a i |-> Frac f (acc m i) *)

  forevery_iso fold_bij
    (fun (i : abs d) -> Cell a i |-> Frac f (acc m i));
  (* forall+ (j : abs (fold_outer d)).
        Cell a (fold_bij.gg j) |-> Frac f (acc m (fold_bij.gg j)) *)

  forevery_map
    (fun (j : abs (fold_outer d)) ->
      Cell a (fold_bij.gg j) |-> Frac f (acc m (fold_bij.gg j)))
    (fun (j : abs (fold_outer d)) ->
      Cell (from_array (tlayout_fold_outer l) (core a)) j
        |-> Frac f (acc (fold_chest m) j))
    fn j {
      tensor_pts_to_cell_eq a (fold_bij.gg j) f (acc m (fold_bij.gg j));
      tensor_pts_to_cell_eq (from_array (tlayout_fold_outer l) (core a)) j f
        (acc (fold_chest m) j);
      rewrite
        Cell a (fold_bij.gg j) |-> Frac f (acc m (fold_bij.gg j))
      as
        Cell (from_array (tlayout_fold_outer l) (core a)) j
          |-> Frac f (acc (fold_chest m) j);
    };

  tensor_implode (from_array (tlayout_fold_outer l) (core a));
}

ghost
fn tensor_unfold_outer
  (#et : Type0)
  (#r: nat {r > 1}) (#d: shape r)
  (#l: tlayout d)
  (a : tensor et (tlayout_fold_outer l))
  (#f: perm) (#m : Chest.t (fold_outer d) et)
  requires
    a |-> Frac f m
  ensures
    from_array l (core a) |-> Frac f (unfold_chest m)
{
  tensor_pts_to_ref a;
  tensor_explode a;
  (* forall+ (j : abs (fold_outer d)). Cell a j |-> Frac f (acc m j) *)

  forevery_iso (bij_sym fold_bij)
    (fun (j : abs (fold_outer d)) -> Cell a j |-> Frac f (acc m j));
  (* forall+ (i : abs d).
        Cell a (fold_bij.ff i) |-> Frac f (acc m (fold_bij.ff i)) *)

  forevery_map
    (fun (i : abs d) ->
      Cell a (fold_bij.ff i) |-> Frac f (acc m (fold_bij.ff i)))
    (fun (i : abs d) ->
      Cell (from_array l (core a)) i
        |-> Frac f (acc (unfold_chest m) i))
    fn i {
      tensor_pts_to_cell_eq a (fold_bij.ff i) f (acc m (fold_bij.ff i));
      tensor_pts_to_cell_eq (from_array l (core a)) i f
        (acc (unfold_chest m) i);
      rewrite
        Cell a (fold_bij.ff i) |-> Frac f (acc m (fold_bij.ff i))
      as
        Cell (from_array l (core a)) i
          |-> Frac f (acc (unfold_chest m) i);
    };

  tensor_implode (from_array l (core a));
}
(* Rank-2 conveniences: explode/implode/ilower/iraise presented over the
   (natlt rows & natlt cols) index pair, as special cases of the generic
   rank-r operations above. *)

inline_for_extraction noextract
unfold
let abs_bij2 (#rows #cols : nat)
  : (abs (rows @| cols @| INil) =~ (natlt rows & natlt cols)) =
  {
    ff = (fun (i, (j, ())) -> (i, j));
    gg = (fun (i, j) -> (i, (j, ())));
  }

ghost
fn tensor_explode2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : tensor et l)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    a |-> Frac f s
  ensures
    forall+ (ij : natlt rows & natlt cols).
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij)))
{
  tensor_explode a;
  forevery_iso #(abs (rows @| cols @| INil)) #(natlt rows & natlt cols)
    abs_bij2 (fun (i : abs (rows @| cols @| INil)) -> Cell a i |-> Frac f (acc s i));
  forevery_ext
    (fun (ij : natlt rows & natlt cols) ->
      Cell a (abs_bij2.gg ij) |-> Frac f (acc s (abs_bij2.gg ij)))
    (fun (ij : natlt rows & natlt cols) ->
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij))));
}

ghost
fn tensor_implode2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : tensor et l)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    pure (SZ.fits (tlayout_ulen l))
  requires
    forall+ (ij : natlt rows & natlt cols).
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij)))
  ensures
    a |-> Frac f s
{
  forevery_iso #(natlt rows & natlt cols) #(abs (rows @| cols @| INil))
    (bij_sym abs_bij2)
    (fun (ij : natlt rows & natlt cols) ->
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij))));
  forevery_ext
    (fun (i : abs (rows @| cols @| INil)) ->
      Cell a (idx2 (fst ((bij_sym abs_bij2).gg i)) (snd ((bij_sym abs_bij2).gg i)))
        |-> Frac f (acc s (idx2 (fst ((bij_sym abs_bij2).gg i)) (snd ((bij_sym abs_bij2).gg i)))))
    (fun (i : abs (rows @| cols @| INil)) -> Cell a i |-> Frac f (acc s i));
  tensor_implode a;
}

ghost
fn tensor_ilower2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : tensor et l)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      Cell a (idx2 r c) |-> Frac f (acc s (idx2 r c)))
{
  tensor_pts_to_ref a;
  tensor_explode2 a;
  forevery_unflatten'
    (fun (ij : natlt rows & natlt cols) ->
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij))));
}

ghost
fn tensor_iraise2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : tensor et l)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      Cell a (idx2 r c) |-> Frac f (acc s (idx2 r c)))
  ensures
    a |-> Frac f s
{
  forevery_flatten'
    (fun (ij : natlt rows & natlt cols) ->
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij))));
  tensor_implode2 a;
}

let ref_of_tensor_cell
  (#et : Type0)
  (#r : nat) (#s : shape r) (#l : tlayout s)
  (a : tensor et l)
  (i : abs s)
  : GTot (ref et)
  = Array.Core.ref_of_array_cell (core a) (l.imap.f i)

inline_for_extraction noextract
fn get_ref_of_tensor_cell
  (#et : Type0)
  (#r : nat) (#s : shape r) (#l : tlayout s)
  (a : tensor et l) {| c : ctlayout l |}
  (i : conc s)
  returns
    r : ref et
  ensures
    pure (r == ref_of_tensor_cell a (up i))
{
  Array.Core.get_ref_of_array_cell (core a) (c.cimap i)
}

ghost
fn tensor_cell_to_ref
  (#et : Type0)
  (#r : nat) (#s : shape r) (#l : tlayout s)
  (a : tensor et l)
  (i : abs s)
  (#f : perm)
  (#v : erased et)
  requires
    Cell a i |-> Frac f v
  ensures
    ref_of_tensor_cell a i |-> Frac f v
{
  tensor_pts_to_cell_eq a i f v;
  rewrite Cell a i |-> Frac f v
       as pts_to_cell (core a) #f (l.imap.f i) v;
  Array.Core.array_cell_to_ref (core a) (l.imap.f i);
}

ghost
fn tensor_cell_from_ref
  (#et : Type0)
  (#r : nat) (#s : shape r) (#l : tlayout s)
  (a : tensor et l)
  (i : abs s)
  (#f : perm)
  (#v : erased et)
  requires
    ref_of_tensor_cell a i |-> Frac f v
  ensures
    Cell a i |-> Frac f v
{
  tensor_pts_to_cell_eq a i f v;
  Array.Core.array_cell_from_ref (core a) (l.imap.f i);
  rewrite pts_to_cell (core a) #f (l.imap.f i) v
       as Cell a i |-> Frac f v;
}
