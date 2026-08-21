module Kuiper.TensorRO
include Kuiper.Shape
include Kuiper.Chest
include Kuiper.Tensor.Layout
#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.Bijection
open Kuiper.Shape
open Kuiper.Chest
open FStar.Tactics.Typeclasses { no_method }
open Pulse.Lib.Trade
open Kuiper.Shareable

module IA = Kuiper.IArray
module IV = Kuiper.IView
module KT = Kuiper.Tensor
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2
module Trade = Pulse.Lib.Trade

(* ------------------------------------------------------------------------ *)
(* Representation.

   A read-only tensor is exactly a base array [larray et l.ulen], viewed
   through the *raw* indexing view over its own [ulen] underlying cells (an
   [IArray] over [IView.raw_view]).  The abstract chest [s : chest d et] is
   linked to the underlying contents [v : natlt ulen -> et] by the (possibly
   non-injective!) layout map [l.imap]:  [acc s i == v (l.imap i)].  Because we
   own each *underlying* cell exactly once (keyed by [natlt ulen], not by the
   abstract index [abs d]), several abstract indices may alias the same cell —
   this is what makes zero-copy broadcast views sound.  We never expose a
   whole-tensor "explode" into per-abstract-cell ownership (that would double
   count aliased cells), nor the injective-only conversions from whole-tensor
   ownership to a family of abstract cell references. *)
(* ------------------------------------------------------------------------ *)

inline_for_extraction noextract
let uview (#r : erased nat) (#d : shape r) (l : vtlayout d) : IV.aiview =
  IV.raw_view #l.ulen

inline_for_extraction noextract
let rotensor (et : Type0) (#r : nat) (#d : shape r) (l : vtlayout d) : Type0 =
  IA.iarray et (uview l)

let is_global
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l) : prop
  = IA.is_global a

inline_for_extraction noextract
let from_array
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (l : vtlayout d)
  (a : larray et (vtlayout_ulen l))
  : rotensor et l
  = IA.from_array (uview l) a

inline_for_extraction noextract
let core
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  : larray et (vtlayout_ulen l)
  = IA.core a

let lem_core_from_array
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]
  = ()

let lem_from_array_core
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (p : larray et (vtlayout_ulen l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let lem_is_global_iff_core
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]
  = ()

(* [(uview l).ait] unfolds to [natlt l.ulen]; give the solver this fact. *)
let uview_ait (#r : erased nat) (#d : shape r) (l : vtlayout d) (_ : unit)
  : Lemma ((uview l).ait == natlt (vtlayout_ulen l))
  = ()

(* The chest determined by an underlying contents function through the layout
   map.  This is the only place the (non-injective) [l.imap] is used to relate
   abstract indices to underlying cells. *)
let ro_chest
  (#et : Type0) (#r : nat) (#d : shape r)
  (l : vtlayout d)
  (v : natlt (vtlayout_ulen l) -> GTot et)
  : chest d et
  = Chest.mk d (fun (i : abs d) -> v (l.imap i))

let tensor_pts_to
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  ([@@@mkey] a : rotensor et l)
  (#[T.exact (`1.0R)] f : perm)
  (s : chest d et)
  : slprop
  = exists* (v : natlt (vtlayout_ulen l) -> GTot et).
      IA.iarray_pts_to a #f v **
      pure (s == ro_chest l v)

(* The raw indexing view over the [ulen] backing cells is concretizable
   whenever the layout is; this lets us reuse [IArray.iarray_read]. *)
inline_for_extraction noextract
let raw_cview (len : erased nat) (pf : squash (SZ.fits len))
  : IV.ciview (IV.raw_view #len)
  = {
      len_fits = pf;
      sch = IV.raw_ciview_schema len;
      step = { cimap = Kuiper.Injection.cinj_id };
    }

inline_for_extraction noextract
instance cvtlayout_ciview
  (#r : erased nat) (#d : shape r)
  (#l : vtlayout d) (cv : cvtlayout l)
  : IV.ciview (uview l)
  = raw_cview l.ulen cv.ulen_fits

instance is_send_across_global_tensor
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l { is_global a })
  (#f : perm) (s : chest d et)
  : is_send_across gpu_of (tensor_pts_to a #f s)
  =
  let pf : is_send_across gpu_of
             (exists* (v : natlt (vtlayout_ulen l) -> GTot et).
                IA.iarray_pts_to a #f v **
                pure (s == ro_chest l v))
         = solve in
  pf

ghost
fn tensor_pts_to_ref
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (#f : perm) (#s : chest d et)
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (vtlayout_ulen l))
{
  unfold tensor_pts_to a #f s;
  with v. assert IA.iarray_pts_to a #f v;
  IA.iarray_pts_to_ref a;
  fold tensor_pts_to a #f s;
}

ghost
fn tensor_pts_to_ref_located
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (#loc : loc_id)
  (#f : perm) (#s : chest d et)
  preserves
    on loc (a |-> Frac f s)
  ensures
    pure (SZ.fits (vtlayout_ulen l))
{
  map_loc loc
    #(a |-> Frac f s)
    #(a |-> Frac f s ** pure (SZ.fits (vtlayout_ulen l)))
  fn _ {
    tensor_pts_to_ref a;
  };
}

ghost
fn tensor_pts_to_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
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
  with v1. assert IA.iarray_pts_to a #f1 v1;
  unfold tensor_pts_to a #f2 s2;
  with v2. assert IA.iarray_pts_to a #f2 v2;
  IA.iarray_pts_to_eq a #f1 f2 #v1 #v2;
  (* now: iarray_pts_to a #f1 v2 ** iarray_pts_to a #f2 v2, with s2 == ro_chest l v2 *)
  fold tensor_pts_to a #f1 s2;
  fold tensor_pts_to a #f2 s2;
}

ghost
fn tensor_share_n
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l) (k : pos)
  (#f : perm) (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s
{
  unfold tensor_pts_to a #f s;
  with v. assert IA.iarray_pts_to a #f v;
  IA.iarray_share_n a k;
  forevery_map
    (fun (i:natlt k) -> IA.iarray_pts_to a #(f /. k) v)
    (fun (i:natlt k) -> tensor_pts_to a #(f /. k) s)
    fn i { fold tensor_pts_to a #(f /. k) s };
}

ghost
fn tensor_gather_n
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l) (k : pos)
  (#f : perm) (#s : chest d et)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s
{
  (* Expose the underlying-cells view of each fractional copy. *)
  forevery_map
    (fun (_:natlt k) -> tensor_pts_to a #(f /. k) s)
    (fun (_:natlt k) ->
       exists* (v : natlt (vtlayout_ulen l) -> GTot et).
         IA.iarray_pts_to a #(f /. k) v ** pure (s == ro_chest l v))
    fn i { unfold tensor_pts_to a #(f /. k) s };
  (* Choose a witness function for each copy. *)
  let vf = forevery_exists
    (fun (i:natlt k) (v : natlt (vtlayout_ulen l) -> GTot et) ->
       IA.iarray_pts_to a #(f /. k) v ** pure (s == ro_chest l v));
  forevery_unzip
    (fun (i:natlt k) -> IA.iarray_pts_to a #(f /. k) (vf i))
    (fun (i:natlt k) -> pure (s == ro_chest l (vf i)));
  forevery_elim_pure (fun (i:natlt k) -> s == ro_chest l (vf i));
  (* All fractional copies describe the *same* physical array, so their
     witness functions must agree; align them to [vf (k-1)] and gather. *)
  forevery_natlt_pop k (fun (i:natlt k) -> IA.iarray_pts_to a #(f /. k) (vf i));
  forevery_map_extra
    (IA.iarray_pts_to a #(f /. k) (vf (k-1)))
    (fun (j:natlt (k-1)) -> IA.iarray_pts_to a #(f /. k) (vf (natlt_coerce j)))
    (fun (j:natlt (k-1)) -> IA.iarray_pts_to a #(f /. k) (vf (k-1)))
    fn j { IA.iarray_pts_to_eq a #(f /. k) (f /. k) #(vf (natlt_coerce j)) #(vf (k-1)); };
  forevery_natlt_push k (fun (i:natlt k) -> IA.iarray_pts_to a #(f /. k) (vf (k-1)));
  IA.iarray_gather_n a k #f #(vf (k-1));
  fold tensor_pts_to a #f s;
}

ghost
fn tensor_gather_n_underspec
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l) (k : pos)
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
  (#l : vtlayout d) {| cv : cvtlayout l |}
  (a : rotensor et l)
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
  let cm : szlt l.ulen = cv.cimap i;
  let res = IA.iarray_read #et #(uview l) #(cvtlayout_ciview cv) a cm;
  (* [(uview l).ait] is definitionally [natlt l.ulen], but the solver no longer
     sees through the projection; state the type equality and coerce. *)
  uview_ait l ();
  assert pure (coerce_eq #_ #(natlt (vtlayout_ulen l)) () (IV.ci_to_ai (uview l) cm)
                 == l.imap (up i));
  fold tensor_pts_to a #f s;
  res
}

let tensor_pts_to_cell
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  ([@@@mkey] a : rotensor et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] i : abs d)
  (v : et)
  : slprop
  = pts_to_cell (core a) #f (l.imap i) v

let tensor_pts_to_cell_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l) (i : abs d) (f : perm) (v : et)
  : Lemma (Cell a i |-> Frac f v
           ==
           pts_to_cell (core a) #f (l.imap i) v)
  = ()

instance is_send_across_global_tensor_cell
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l { is_global a })
  (#f : perm) (i : abs d) (v : et)
  : is_send_across gpu_of (tensor_pts_to_cell a #f i v)
  =
  let pf : is_send_across gpu_of (pts_to_cell (core a) #f (l.imap i) v) = solve in
  pf

inline_for_extraction noextract
fn tensor_read_cell
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : vtlayout d) {| cv : cvtlayout l |}
  (a : rotensor et l)
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
  tensor_pts_to_cell_eq a (up i) f s;
  rewrite (Cell a (up i) |-> Frac f s)
       as (pts_to_cell (core a) #f (l.imap (up i)) s);
  let cm : SZ.t = cv.cimap i;
  let res = slice_read (core a) cm;
  rewrite (pts_to_cell (core a) #f (l.imap (up i)) s)
       as (Cell a (up i) |-> Frac f s);
  res
}

(* ---------------------------- Slicing (layout) ---------------------------- *)

inline_for_extraction noextract
let ctlayout_slice_cimap
  (#n : erased nat) (d : shape n) (l : vtlayout d)
  {| c : cvtlayout l |}
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
      l.imap (up ((c_conc_bring_forward_bij i d).cgg (j, idx)));
      == { bring_forward_commute2 i d j idx }
      l.imap ((abs_bring_forward_bij i d).gg (SZ.v j, up idx));
      == {}
      tlayout_slice_imap d l i j (up idx);
    };
    res

inline_for_extraction noextract
instance ctlayout_slice
  (#n : erased nat) (#d : shape n) (l : vtlayout d)
  {| cvtlayout l |}
  (i : erased nat{i < n}) (j : erased nat{j < (d @! i)})
  {| ix : concrete_sz i |} {| jx : concrete_sz j |}
  (#r' : erased nat) (#d' : shape r')
  (#_ : reveal r' == n-1)
  (#_ : d' == modulo_i i d)
  : cvtlayout #r' #d' (vtlayout_slice l i j) =
  {
    ulen_fits = ();
    all_fit = ();
    cimap = (fun idx ->
      ctlayout_slice_cimap d l (concr' ix) (concr' jx) idx);
  }

inline_for_extraction noextract
let sliceof
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : rotensor et (vtlayout_slice l i j)
  = from_array (vtlayout_slice l i j) (core a)

let lem_sliceof_core
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (i : erased nat{i < r}) (j : erased nat{j < d @! i})
  : Lemma (core (sliceof a i j) == core a)
          [SMTPat (sliceof a i j)]
  = ()

let lem_is_global_iff_sliceof
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (i : natlt r) (j : natlt (d @! i))
  : Lemma (ensures is_global (sliceof a i j) <==> is_global a)
          [SMTPat (is_global (sliceof a i j))]
  = ()

#push-options "--warn_error -271"
let tensor_slice_cell_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (i : natlt r) (j : natlt (d @! i))
  (k : abs (modulo_i i d)) (f : perm) (v : et)
  : Lemma (Cell (sliceof a i j) k |-> Frac f v
           ==
           Cell a ((abs_bring_forward_bij i d).gg (j, k)) |-> Frac f v)
           [SMTPat (Cell (sliceof a i j) k |-> Frac f v)]
  = ()
#pop-options

(* ---------------------------- Bijections ---------------------------- *)

inline_for_extraction noextract
instance ctlayout_bij
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2 { all_fit d2 })
  (f : abs d1 =~ abs d2)
  (fconc: conc d2 -> conc d1)
  (fconc_correct: (x: conc d2) -> up (fconc x) == f.gg (up x))
  (l : vtlayout d1) {| c: cvtlayout l |}
  : cvtlayout #r2 #d2 (vtlayout_bij f l) =
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
  (#l : vtlayout d1)
  (a : rotensor et l)
  (#fp : perm) (#m : Chest.t d1 et)
  requires
    a |-> Frac fp m
  ensures
    from_array (vtlayout_bij f l) (core a) |-> Frac fp (Chest.mk d2 (fun i -> Chest.acc m (i <~| f)))
{
  unfold tensor_pts_to a #fp m;
  with v. assert IA.iarray_pts_to a #fp v;
  rewrite (IA.iarray_pts_to a #fp v)
       as (IA.iarray_pts_to (from_array (vtlayout_bij f l) (core a)) #fp v);
  assert pure (Kuiper.Chest.equal
                 (Chest.mk d2 (fun i -> Chest.acc m (i <~| f)))
                 (ro_chest (vtlayout_bij f l) v));
  fold tensor_pts_to (from_array (vtlayout_bij f l) (core a)) #fp
       (Chest.mk d2 (fun i -> Chest.acc m (i <~| f)));
}

let lem_ro_chest_bij_index
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (l : vtlayout d1)
  (v : natlt l.ulen -> GTot et)
  (m : chest d1 et)
  (i : abs d1)
  : Lemma
      (requires
        Chest.mk d2 (fun j -> Chest.acc m (j <~| f))
          == ro_chest (vtlayout_bij f l) v)
      (ensures Chest.acc m i == Chest.acc (ro_chest l v) i)
      [SMTPat (Chest.acc m i);
       SMTPat (Chest.acc (ro_chest l v) i);
       SMTPat (f.ff i)]
  = (* Spell out each step: the solver no longer gets the intermediate ground
       terms from the sibling conjuncts of a merged query. *)
    f.gg_ff i;
    assert (
      Chest.acc (Chest.mk d2 (fun j -> Chest.acc m (j <~| f))) (f.ff i)
        == Chest.acc (ro_chest (vtlayout_bij f l) v) (f.ff i));
    assert (
      Chest.acc (Chest.mk d2 (fun j -> Chest.acc m (j <~| f))) (f.ff i)
        == Chest.acc m (f.ff i <~| f));
    assert (f.ff i <~| f == i);
    assert (
      Chest.acc (ro_chest (vtlayout_bij f l) v) (f.ff i)
        == v ((vtlayout_bij f l).imap (f.ff i)));
    assert ((vtlayout_bij f l).imap (f.ff i) == l.imap (f.gg (f.ff i)));
    assert (Chest.acc (ro_chest l v) i == v (l.imap i));
    ()

let lem_ro_chest_bij
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (l : vtlayout d1)
  (v : natlt l.ulen -> GTot et)
  (m : chest d1 et)
  : Lemma
      (requires
        Chest.mk d2 (fun j -> Chest.acc m (j <~| f))
          == ro_chest (vtlayout_bij f l) v)
      (ensures Chest.equal m (ro_chest l v))
    (* The precondition of [lem_ro_chest_bij_index] does not depend on [i] and
       holds here, so introduce the quantifier directly, giving the motive
       explicitly (it is no longer inferable from the ambient query). *)
  = FStar.Classical.forall_intro
      #(abs d1)
      #(fun i -> Chest.acc m i == Chest.acc (ro_chest l v) i)
      (fun i -> lem_ro_chest_bij_index f l v m i);
    assert (forall i. Chest.acc m i == Chest.acc (ro_chest l v) i);
    Chest.lemma_equal_intro m (ro_chest l v)

ghost
fn tensor_unapply_bij
  (#et : Type0)
  (#r1 : nat) (#d1 : shape r1)
  (#r2 : nat) (#d2 : shape r2)
  (f : abs d1 =~ abs d2)
  (#l : vtlayout d1)
  (a : rotensor et l)
  (#fp : perm) (#m : Chest.t d1 et)
  requires
    from_array (vtlayout_bij f l) (core a) |->
      Frac fp (Chest.mk d2 (fun i -> Chest.acc m (i <~| f)))
  ensures
    a |-> Frac fp m
{
  unfold tensor_pts_to
    (from_array (vtlayout_bij f l) (core a)) #fp
    (Chest.mk d2 (fun i -> Chest.acc m (i <~| f)));
  with v. assert
    IA.iarray_pts_to (from_array (vtlayout_bij f l) (core a)) #fp v;
  rewrite
    (IA.iarray_pts_to (from_array (vtlayout_bij f l) (core a)) #fp v)
    as (IA.iarray_pts_to a #fp v);
  lem_ro_chest_bij f l v m;
  fold tensor_pts_to a #fp m;
}

inline_for_extraction noextract
let cvtlayout_extended_cimap
  (#r : erased nat) (#d : shape r)
  (l : vtlayout d) {| c : cvtlayout l |}
  (e : erased nat{SZ.fits e})
  (i : conc (e @| d))
  : Tot (x : SZ.t{SZ.v x == (extended_layout l e).imap (up i)})
  =
    let (_, rest) = i in
    c.cimap rest

inline_for_extraction noextract
instance cvtlayout_extended
  (#r : erased nat) (#d : shape r)
  (l : vtlayout d) {| c : cvtlayout l |}
  (e : erased nat{SZ.fits e})
  : cvtlayout (extended_layout l e)
  = Mkcvtlayout #(r + 1) #(e @| d) #(extended_layout l e)
      c.ulen_fits () (cvtlayout_extended_cimap l e)

let lem_ro_chest_extended_index
  (#et : Type0)
  (#r : nat) (#d : shape r)
  (l : vtlayout d)
  (v : natlt l.ulen -> GTot et)
  (m : chest d et)
  (e : pos)
  (i : abs d)
  : Lemma
      (requires
        Chest.mk (e @| d) (function (_, j) -> Chest.acc m j)
          == ro_chest (extended_layout l e) v)
      (ensures Chest.acc m i == Chest.acc (ro_chest l v) i)
      [SMTPat (Chest.acc m i);
       SMTPat (Chest.acc (ro_chest l v) i);
       SMTPat (Chest.acc (ro_chest (extended_layout l e) v) (0, i))]
  = (* Spell out each step: the solver no longer gets the intermediate ground
       terms from the sibling conjuncts of a merged query. *)
    assert (
      Chest.acc
        (Chest.mk (e @| d) (function (_, j) -> Chest.acc m j))
        (0, i)
        == Chest.acc (ro_chest (extended_layout l e) v) (0, i));
    assert (
      Chest.acc
        (Chest.mk (e @| d) (function (_, j) -> Chest.acc m j))
        (0, i)
        == Chest.acc m i);
    assert (
      Chest.acc (ro_chest (extended_layout l e) v) (0, i)
        == v ((extended_layout l e).imap (0, i)));
    assert ((extended_layout l e).imap (0, i) == l.imap i);
    assert (Chest.acc (ro_chest l v) i == v (l.imap i));
    ()

let lem_ro_chest_extended
  (#et : Type0)
  (#r : nat) (#d : shape r)
  (l : vtlayout d)
  (v : natlt l.ulen -> GTot et)
  (m : chest d et)
  (e : pos)
  : Lemma
      (requires
        Chest.mk (e @| d) (function (_, j) -> Chest.acc m j)
          == ro_chest (extended_layout l e) v)
      (ensures Chest.equal m (ro_chest l v))
    (* Motive given explicitly; the precondition does not depend on [i]. *)
  = FStar.Classical.forall_intro
      #(abs d)
      #(fun i -> Chest.acc m i == Chest.acc (ro_chest l v) i)
      (fun i -> lem_ro_chest_extended_index l v m e i);
    assert (forall i. Chest.acc m i == Chest.acc (ro_chest l v) i);
    Chest.lemma_equal_intro m (ro_chest l v)

ghost
fn rotensor_add_dim
  (#et : Type0)
  (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (#f : perm) (#m : Chest.t d et)
  (e : nat)
  requires
    a |-> Frac f m
  ensures
    from_array (extended_layout l e) (core a) |-> Frac f (Chest.mk (e @| d) (function (_, i) -> acc m i))
{
  unfold tensor_pts_to a #f m;
  with v. assert IA.iarray_pts_to a #f v;
  rewrite (IA.iarray_pts_to a #f v)
       as (IA.iarray_pts_to (from_array (extended_layout l e) (core a)) #f v);
  assert pure (Kuiper.Chest.equal
                 (Chest.mk (e @| d) (function (_, i) -> acc m i))
                 (ro_chest (extended_layout l e) v));
  fold tensor_pts_to (from_array (extended_layout l e) (core a)) #f
       (Chest.mk (e @| d) (function (_, i) -> acc m i));
}

(* Value-returning counterpart of [rotensor_add_dim].  The new dimension [e] is
   spec-only: the returned backing array does not depend on it -- its underlying
   length is unchanged, [extended_layout] only grows the *erased* shape -- so [e]
   is [erased].  This is what lets a caller feed a ghost dimension to this
   *stateful* view: with a concrete [nat] parameter the [reveal] coercion at the
   call site would give the stateful application a ghost effect, which is
   rejected (Error 228).  Inside,
   [from_array (extended_layout l e) (core a)] stays [Tot] -- the [reveal] is
   absorbed into the erasable [vtlayout] -- so [b] is a genuine non-ghost value
   (the identity on the backing pointer). *)
inline_for_extraction noextract
fn rotensor_add_dim_view
  (#et : Type0)
  (#r : erased nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (#f : perm) (#m : Chest.t d et)
  (e : erased nat)
  requires
    a |-> Frac f m
  returns
    b : rotensor et (extended_layout l e)
  ensures
    rewrites_to b (from_array (extended_layout l e) (core a)) **
    b |-> Frac f (Chest.mk (e @| d) (function (_, i) -> acc m i))
{
  rotensor_add_dim a e;
  let b = from_array (extended_layout l e) (core a);
  assert rewrites_to b (from_array (extended_layout l e) (core a));
  b
}

ghost
fn rotensor_remove_dim
  (#et : Type0)
  (#r : nat) (#d : shape r)
  (#l : vtlayout d)
  (a : rotensor et l)
  (#f : perm) (#m : Chest.t d et)
  (e : pos)
  requires
    from_array (extended_layout l e) (core a) |->
      Frac f (Chest.mk (e @| d) (function (_, i) -> acc m i))
  ensures
    a |-> Frac f m
{
  unfold tensor_pts_to
    (from_array (extended_layout l e) (core a)) #f
    (Chest.mk (e @| d) (function (_, i) -> acc m i));
  with v. assert
    IA.iarray_pts_to (from_array (extended_layout l e) (core a)) #f v;
  rewrite
    (IA.iarray_pts_to (from_array (extended_layout l e) (core a)) #f v)
    as (IA.iarray_pts_to a #f v);
  lem_ro_chest_extended l v m e;
  fold tensor_pts_to a #f m;
}

(* ------------------------------------------------------------------------ *)
(* Ownership-safe conversion  Tensor <-> TensorRO.                          *)
(* ------------------------------------------------------------------------ *)

let lem_vtlayout_of_tlayout_ulen (#r : nat) (#d : shape r) (l : tlayout d)
  : Lemma (ensures vtlayout_ulen (vtlayout_of_tlayout l) == tlayout_ulen l)
          [SMTPat (vtlayout_ulen (vtlayout_of_tlayout l))]
  = ()

inline_for_extraction noextract
instance cvtlayout_of_ctlayout
  (#r : erased nat) (#d : shape r)
  (l : tlayout d) {| c : ctlayout l |}
  : cvtlayout (vtlayout_of_tlayout l)
  = {
      ulen_fits = c.ulen_fits;
      all_fit = c.all_fit;
      cimap = (fun (i : conc d) -> c.cimap i);
    }

ghost
fn tensor_borrow_ro
  (#et : Type0)
  (#r : nat) (#d : shape r)
  (#l : tlayout d { is_full l })
  (t : KT.tensor et l)
  (#f : perm) (#s : chest d et)
  requires
    KT.tensor_pts_to t #f s
  ensures
    factored
      (rotensor_of_tensor t |-> Frac f s)
      (KT.tensor_pts_to t #f s)
{
  let vl : vtlayout d = vtlayout_of_tlayout l;
  let ro : rotensor et vl = from_array vl (KT.core t);
  (* Concretize the writable tensor to its backing array, then re-view it as a
     read-only tensor with the (faithful) same-index-map layout. *)
  KT.tensor_concr t;
  IA.iarray_begin_ (KT.core t);
  rewrite (IA.iarray_pts_to (IA.from_array (IV.raw_view #l.ulen) (KT.core t)) #f (IA.g_seq_acc (to_seq l s)))
       as (IA.iarray_pts_to ro #f (IA.g_seq_acc (to_seq l s)));
  assert pure (Kuiper.Chest.equal s (ro_chest vl (IA.g_seq_acc (to_seq l s))));
  fold tensor_pts_to ro #f s;
  (* Recovery trade: give back the read-only view, recover the writable tensor
     with its original contents (a read-only view never mutates the store). *)
  Trade.intro_trade
    (tensor_pts_to ro #f s)
    (KT.tensor_pts_to t #f s)
    emp
    fn _ {
       unfold tensor_pts_to ro #f s;
       with v. assert (IA.iarray_pts_to ro #f v);
       IA.iarray_end_ ro;
       assert pure (Seq.equal (Seq.init_ghost l.ulen v) (to_seq l s));
       rewrite (IA.core ro |-> Frac f (Seq.init_ghost l.ulen v))
            as (KT.core t |-> Frac f (to_seq l s));
       KT.tensor_abs l (KT.core t);
       rewrite (KT.tensor_pts_to (KT.from_array l (KT.core t)) #f s)
            as (KT.tensor_pts_to t #f s);
    };
  rewrite each ro as (rotensor_of_tensor t);
}

(* Ergonomic wrapper: same proof, but the view is returned.  Not [ghost] since
   [rotensor] is informative; it extracts to nothing but the identity on the
   backing pointer ([from_array]/[core] are inline and the layouts erased). *)
inline_for_extraction noextract
fn tensor_to_rotensor
  (#et : Type0)
  (#r : erased nat) (#d : shape r)
  (#l : tlayout d { is_full l })
  (t : KT.tensor et l)
  (#f : perm) (#s : chest d et)
  requires
    KT.tensor_pts_to t #f s
  returns
    a : rotensor et (vtlayout_of_tlayout l)
  ensures
    rewrites_to a (rotensor_of_tensor t) **
    factored
      (a |-> Frac f s)
      (KT.tensor_pts_to t #f s)
{
  tensor_borrow_ro t;
  let a = rotensor_of_tensor t;
  assert rewrites_to a (rotensor_of_tensor t);
  a
}

ghost
fn rotensor_to_tensor
  (#et : Type0)
  (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (t : KT.tensor et l)
  (a : rotensor et (vtlayout_of_tlayout l))
  (#f : perm) (#s : chest d et)
  requires
    factored
      (a |-> Frac f s)
      (KT.tensor_pts_to t #f s)
  ensures
    KT.tensor_pts_to t #f s
{
  Trade.elim_trade (a |-> Frac f s) (KT.tensor_pts_to t #f s);
}
