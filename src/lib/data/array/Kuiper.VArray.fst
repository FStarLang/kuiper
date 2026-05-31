module Kuiper.VArray
#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.Injection
module B = Kuiper.Array (* base *)
module T = FStar.Tactics.V2
module SZ = Kuiper.SizeT
module IView = Kuiper.IView
module IArray = Kuiper.IArray

inline_for_extraction
type varray
  (#et:Type0) (#st : Type0)
  (vw : aview et st)
= IArray.iarray et vw.iview

let is_global
  (#et:Type0) (#st : Type0)
  (#vw : aview et st)
  (arr: varray vw)
: prop
= IArray.is_global arr

inline_for_extraction noextract
let from_array
  (#a : Type0) (#st : Type0)
  (vw : aview a st)
  (arr : larray a (len vw))
  : varray vw
  = IArray.from_array vw.iview arr

let core a = IArray.core a

let lem_is_global_iff_core
  (#a : Type0)
  (#st : Type) (#vw : aview a st)
  (g : varray vw)
  : Lemma (ensures is_global g <==> is_global_array (core g))
          [SMTPat (is_global g)]
  = ()

let lem_from_array_core
  (#a : Type0)
  (#st : Type0) (#vw : aview a st)
  (arr : varray vw)
  : Lemma (ensures from_array vw (core arr) == arr)
          [SMTPat (core arr)]
  = ()

let lem_core_from_array
  (#a : Type0)
  (#st : Type0) (#vw : aview a st)
  (p : larray a (len vw))
  : Lemma (ensures core (from_array vw p) == p)
          [SMTPat (from_array vw p)]
  = ()

let varray_pts_to_cell
  (#et:Type0) (#st:Type0)
  (#vw : aview et st)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.iview.ait)
  (v : et)
  : slprop
  = Cell a i |-> Frac f v

let varray_pts_to_cell_eq
  (#et:Type)
  (#st:Type0) (#vw : aview et st)
  (a : varray vw)
  (i : vw.iview.ait)
  (f : perm)
  (v : et)
  : Lemma (varray_pts_to_cell a #f i v
           ==
           pts_to_cell (core a) #f (vw.iview.step.imap.f i) v)
  = IArray.iarray_pts_to_cell_def #et #vw.iview a #f i v

let varray_pts_to
  (#et:Type0) (#st:_) (#vw : aview et st)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : st)
  : slprop
  =
    a |-> Frac f (vw.ctn.acc v)

instance is_send_across_global_varray
  (#et:Type0)
  (#st : Type0)
  (#vw : aview et st)
  (x: varray vw { is_global x })
  (#f : perm)
  (v : st)
  : is_send_across gpu_of (varray_pts_to x #f v)
  = solve

instance is_send_across_global_varray_cell
  (#et:Type0)
  (#st : Type0)
  (#vw : aview et st)
  (a : varray vw { is_global a })
  (#f : perm)
  (i : vw.iview.ait)
  (v : et)
  : is_send_across gpu_of (varray_pts_to_cell a #f i v)
  = solve

ghost
fn varray_pts_to_ref
  (#t:Type0)
  (#st:Type0)
  (#vw : aview t st)
  (a : varray vw)
  (#f : perm)
  (#v : erased st)
  preserves
    a |-> Frac f v
  ensures
    pure (SZ.fits (len vw))
{
  unfold varray_pts_to a #f v;
  IArray.iarray_pts_to_ref a;
  fold varray_pts_to a #f v;
  ()
}

ghost
fn varray_explode
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  (a : varray vw)
  (#f : perm)
  (#v : st)
  requires
    a |-> Frac f v
  ensures
    forall+ (i : vw.iview.ait).
      Cell a i |-> Frac f (vw.ctn.acc v i)
{
  unfold varray_pts_to a #f v;
  IArray.iarray_explode a;
  ()
}

ghost
fn varray_implode
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  (a : varray vw)
  (#f : perm)
  (#v : st)
  requires
    pure (SZ.fits (len vw))
  requires
    forall+ (i : vw.iview.ait).
      Cell a i |-> Frac f (vw.ctn.acc v i)
  ensures
    a |-> Frac f v
{
  IArray.iarray_implode a;
  fold varray_pts_to a #f v;
}

(* Note how the spec type does not change at all. The mapping
is hidden in the view. *)
ghost
fn varray_reindex_
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  (#ait' : Type)
  (bij : vw.iview.ait =~ ait')
  (a : varray vw)
  (#f : perm)
  (#v : st)
  requires
    a |-> Frac f v
  ensures
    from_array (reindex_view vw bij) (core a) |-> Frac f v
{
  unfold varray_pts_to a #f v;
  IArray.iarray_reindex_ bij a;
  rewrite each
    IArray.from_array #et (IView.reindex_view vw.iview #ait' bij) (IArray.core a)
  as
    (from_array (reindex_view vw bij) (core a));
  IArray.iarray_ext (from_array (reindex_view vw bij) (core a))
    _
    ((reindex_view vw bij).ctn.acc v);
  fold varray_pts_to (from_array (reindex_view vw bij) (core a)) #f v;
  ()
}

inline_for_extraction noextract
fn varray_reindex
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  (#ait' : Type)
  (bij : vw.iview.ait =~ ait')
  (a : varray vw)
  (#f : perm)
  (#v : erased st)
  requires
    a |-> Frac f v
  returns
    va : varray (reindex_view vw bij)
  ensures
    va |-> Frac f v **
    pure (core va == core a)
{
  varray_reindex_ bij a;
  from_array (reindex_view vw bij) (core a)
}

ghost
fn varray_review_
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  (#st' : Type)
  (bij : st =~ st')
  (a : varray vw)
  (#f : perm)
  (#v : st)
  requires
    a |-> Frac f v
  ensures
    from_array (review_view vw bij) (core a) |-> Frac f (bij.ff v)
{
  unfold varray_pts_to a #f v;
  IArray.iarray_ext a
    (vw.ctn.acc v)
    ((review_view vw bij).ctn.acc (bij.ff v));

  rewrite IArray.iarray_pts_to #et #vw.iview a #f ((review_view vw bij).ctn.acc (bij.ff v))
       as IArray.iarray_pts_to #et #vw.iview
            (from_array (review_view vw bij) (core a))
            #f
            ((review_view vw bij).ctn.acc (bij.ff v));
  assert pure (vw.iview == (review_view vw bij).iview);
  rewrite IArray.iarray_pts_to #et #vw.iview
            (from_array (review_view vw bij) (core a))
            #f
            ((review_view vw bij).ctn.acc (bij.ff v))
       as IArray.iarray_pts_to #et #(review_view vw bij).iview
            (from_array (review_view vw bij) (core a))
            #f
            ((review_view vw bij).ctn.acc (bij.ff v))
       by slprop_equiv_norm (); // ugly
  fold varray_pts_to (from_array (review_view vw bij) (core a)) #f (bij.ff v);
  ()
}

inline_for_extraction noextract
fn varray_review
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  (#st' : Type)
  (bij : st =~ st')
  (a : varray vw)
  (#f : perm)
  (#v : erased st)
  requires
    a |-> Frac f v
  returns
    va : varray (review_view vw bij)
  ensures
    (va |-> Frac f (bij.ff v)) **
    pure (core va == core a)
{
  varray_review_ bij a;
  from_array (review_view vw bij) (core a)
}


(* Begin viewing something abstractly, with the trivial view. The spec
type are sequences. *)
ghost
fn varray_begin_
  (#et : Type) (#len : erased nat)
  (a : larray et len)
  (#f : perm)
  (#v : lseq et len)
  requires
    a |-> Frac f v
  ensures
    from_array (raw_view #et #len) a |-> Frac f v
{
  IArray.iarray_begin_ a;
  rewrite each IArray.from_array (IView.raw_view #len) a as
    (from_array (raw_view #et #len) a);
  IArray.iarray_ext _
    (IArray.g_seq_acc v)
    ((raw_view #et #len).ctn.acc v);
  fold varray_pts_to (from_array (raw_view #et #len) a) #f v;
  ()
}

inline_for_extraction noextract
fn varray_begin
  (#et : Type) (#len : erased nat)
  (a : larray et len)
  (#f : perm)
  (#v : erased (lseq et len))
  requires
    a |-> Frac f v
  returns
    va : varray (raw_view #et #len)
  ensures
    va |-> Frac f v **
    pure (core va == a)
{
  varray_begin_ a;
  from_array (raw_view #et #len) a
}

ghost
fn varray_end_
  (#et : Type) (#len : erased nat)
  (a : varray (raw_view #et #len))
  (#f : perm)
  (#v : lseq et len)
  requires
    a |-> Frac f v
  ensures
    core a |-> Frac f v
{
  unfold varray_pts_to a #f v;
  IArray.iarray_end_ #_ #len a;
  with vv.
    assert pts_to_slice (core a) #f 0 len vv;
    assert (pure (Seq.equal vv v));
  ()
}

inline_for_extraction noextract
fn varray_end
  (#et : Type0) (#len : erased nat)
  (a : varray (raw_view #et #len))
  (#f : perm)
  (#v : erased (lseq et len))
  requires
    a |-> Frac f v
  returns
    a' : larray et len
  ensures
    a' |-> Frac f v
{
  varray_end_ a;
  (core a);
}

ghost
fn varray_cell_reindex
  (#et:Type0) (#st #st':Type0)
  (#f : perm)
  (#vw : aview et st)
  (#vw' : aview et st')
  (a : varray vw)
  (i : vw.iview.ait)
  (a' : varray vw')
  (i' : vw'.iview.ait)
  (#v : et)
  requires
    pure (len vw == len vw' /\ core a == core a')
  requires
    pure (IView.it_to_nat vw.iview i == IView.it_to_nat vw'.iview i')
  requires
    Cell a i |-> Frac f v
  ensures
    Cell a' i' |-> Frac f v
{
  unfold varray_pts_to_cell a #f i v;
  IArray.iarray_cell_reindex a i a' i';
  fold varray_pts_to_cell a' #f i' v;
}

ghost
fn varray_abs
  (#et : Type0) (#st : Type0)
  (vw : aview et st { is_full_view vw })
  (a : larray et (len vw))
  (#f : perm)
  (#v : st)
  requires
    a |-> Frac f (to_seq vw v)
  ensures
    from_array vw a |-> Frac f v
{
  varray_begin_ a;
  let vw0 = raw_view #et #(len vw); rewrite each raw_view #et #(len vw) as vw0;
  let a0 = from_array vw0 a; rewrite each from_array vw0 a as a0;
  varray_pts_to_ref a0;
  assert varray_pts_to a0 #f (to_seq vw v);
  let bij: (vw0.iview.ait =~ vw.iview.ait) =
    bij_sym (Kuiper.IView.full_view_bij vw.iview);
  varray_reindex_ bij a0;
  let vw1 = reindex_view vw0 bij; rewrite each reindex_view vw0 bij as vw1;
  let a1 = from_array vw1 (core a0); rewrite each from_array vw1 (core a0) as a1;
  varray_explode a1;
  forevery_map'
    (fun (i: vw1.iview.ait) ->
      varray_pts_to_cell a1 #f i (vw1.ctn.acc (to_seq vw v) i))
    (fun (i: vw.iview.ait) ->
      varray_pts_to_cell (from_array vw a) #f i (vw.ctn.acc v i))
    fn i i' {
      varray_cell_reindex a1 i (from_array vw a) i';
      rewrite each vw1.ctn.acc (to_seq vw v) i
                as vw.ctn.acc v i';
    };
  varray_implode (from_array vw a) #f;
}

ghost
fn varray_abs'
  (#et : Type0) (#st : Type0)
  (vw : aview et st { is_full_view vw })
  (a : larray et (len vw))
  (#f : perm)
  (#v : lseq et (len vw))
  requires
    a |-> Frac f v
  ensures
    from_array vw a |-> Frac f (from_seq vw v)
{
  rewrite each v as to_seq vw (from_seq vw v);
  varray_abs vw a;
}

ghost
fn varray_abs_alt'
  (#et : Type0) (#st : Type0)
  (vw : aview et st { is_full_view vw })
  (sz : nat { sz == len vw })
  (a : larray et sz)
  (#f : perm)
  (#v : lseq et sz)
  requires
    a |-> Frac f v
  ensures
    from_array vw a |-> Frac f (from_seq vw v)
{
  varray_abs' vw a;
}

ghost
fn varray_concr
  (#et : Type0) (#st : Type0)
  (#vw : aview et st { is_full_view vw })
  (a : varray vw)
  (#f : perm)
  (#v : erased st)
  requires
    a |-> Frac f v
  ensures
    core a |-> Frac f (to_seq vw v)
{
  varray_explode a;
  forevery_map
    #vw.iview.ait
    (fun i -> varray_pts_to_cell a #f i (vw.ctn.acc v i))
    (fun i -> pts_to_cell (core a) #f (vw.iview.step.imap.f i) (vw.ctn.acc v i))
    fn i {
      unfold varray_pts_to_cell a #f i (vw.ctn.acc v i);
      IArray.iarray_pts_to_cell_def a #f i (vw.ctn.acc v i);
      rewrite IArray.iarray_pts_to_cell a #f i (vw.ctn.acc v i)
            as pts_to_cell (core a) #f (vw.iview.step.imap.f i) (vw.ctn.acc v i);
      ()
  };

  (* We now have separate ownership of each cell. Change the index type to natlt. *)
  let bij : (vw.iview.ait =~ natlt (len vw)) = Kuiper.IView.full_view_bij vw.iview;
  forevery_iso bij _;

  (* Now, we need them to be in sequential order, so can use unslice_1. (This can probably
     be done in one step. *)
  assert pure (Functions.is_surj vw.iview.step.imap.f);
  let inv_f : (natlt (len vw) @~> vw.iview.ait) = Injection.inverse' vw.iview.step.imap;
  let perm : (natlt (len vw) =~ natlt (len vw)) = bij_inj' inv_f `bij_comp` bij;
  forevery_iso perm _;

  (* By carefully choosing that permutation, things simplify away. *)
  forevery_map
    #(natlt (len vw))
    (fun i -> pts_to_cell (core a) #f (vw.iview.step.imap.f (bij.gg (perm.gg i))) (vw.ctn.acc v (bij.gg (perm.gg i))))
    (fun i -> pts_to_cell (core a) #f i (to_seq vw v @! i))
    fn i {
      rewrite each vw.iview.step.imap.f (bij.gg (perm.gg i)) as i;
      rewrite each vw.ctn.acc v (bij.gg (perm.gg i)) as (to_seq vw v @! i);
      ()
  };

  array_unslice_1 (core a) #f #(to_seq vw v);
  ()
}

ghost
fn varray_iconcr
  (#et : Type0) (#st : Type0)
  (#vw : aview et st)
  (a : varray vw)
  (#f : perm)
  (#v : erased st)
  requires
    a |-> Frac f v
  ensures
    pure (SZ.fits (len vw)) **
    (forall+ (i : vw.iview.ait).
      pts_to_cell (core a) #f (vw.iview.step.imap.f i) (vw.ctn.acc v i))
{
  unfold varray_pts_to a #f v;
  IArray.iarray_pts_to_ref a;
  IArray.iarray_explode a;
  ghost
  fn aux (i : vw.iview.ait)
    requires
      Cell a i |-> Frac f (vw.ctn.acc v i)
    ensures
      pts_to_cell (core a) #f (vw.iview.step.imap.f i) (vw.ctn.acc v i)
  {
    IArray.iarray_pts_to_cell_def a #f i (vw.ctn.acc v i);
    rewrite
      Cell a i |-> Frac f (vw.ctn.acc v i)
    as
      pts_to_cell (core a) #f (vw.iview.step.imap.f i) (vw.ctn.acc v i);
  };
  forevery_map _ _ aux;
  ()
}

ghost
fn varray_iabs
  (#et : Type0) (#st : Type0)
  (#vw : aview et st)
  (a : varray vw)
  (#f : perm)
  (#v : erased st)
  requires
    pure (SZ.fits (len vw)) **
    (forall+ (i : vw.iview.ait).
      pts_to_cell (core a) #f (vw.iview.step.imap.f i) (vw.ctn.acc v i))
  ensures
    a |-> Frac f v
{
  ghost
  fn aux (i : vw.iview.ait)
    requires
      pts_to_cell (core a) #f (vw.iview.step.imap.f i) (vw.ctn.acc v i)
    ensures
      Cell a i |-> Frac f (vw.ctn.acc v i)
  {
    IArray.iarray_pts_to_cell_def a #f i (vw.ctn.acc v i);
    rewrite
      pts_to_cell (core a) #f (vw.iview.step.imap.f i) (vw.ctn.acc v i)
    as
      Cell a i |-> Frac f (vw.ctn.acc v i);
  };
  forevery_map _ _ aux;
  IArray.iarray_implode a;
  fold varray_pts_to a #f v;
}

inline_for_extraction noextract
fn varray_alloc0
  (#et : Type0) {| sized et |} (len : sz { len > 0 }) (#st : Type0)
  (vw : aview et st { is_full_view vw /\ Len.len vw == len})
  preserves
    cpu
  requires
    pure (SZ.fits len)
  returns
    a : varray vw
  ensures
    exists* v. on gpu_loc (a |-> v)
  ensures
    pure (is_global a) **
    pure (is_full_array (core a))
{
  let a = B.gpu_array_alloc #et len;
  with s. assert on gpu_loc (a |-> s);
  map_loc gpu_loc (fun () -> varray_abs_alt' vw _ a #1.0R #s);
  let r = from_array vw a; assert rewrites_to r (from_array vw a);
  r
}

inline_for_extraction noextract
fn varray_free
  (#et : Type0) (#st : Type0)
  (#vw : aview et st { is_full_view vw })
  (a : varray vw)
  (#v : erased st)
  preserves
    cpu
  requires
    pure (is_full_array (core a)) **
    on gpu_loc (a |-> v)
  ensures emp
{
  map_loc gpu_loc (fun () -> varray_concr a);
  B.gpu_array_free (core a);
}

ghost
fn varray_view_equiv_
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  (a : varray vw)
  (vw' : aview et st { view_equiv vw vw' })
  (#f : perm)
  (#v : erased st)
  requires
    a |-> Frac f v
  ensures
    from_array vw' (core a) |-> Frac f v
{
  varray_pts_to_ref a;
  varray_explode a;
  forevery_map'
    (fun (i: vw.iview.ait) -> varray_pts_to_cell a #f i (vw.ctn.acc v i))
    (fun (i: vw'.iview.ait) -> varray_pts_to_cell (from_array vw' (core a)) #f i (vw'.ctn.acc v i))
    fn i i' {
      varray_cell_reindex a i (from_array vw' (core a)) i';
      rewrite each vw.ctn.acc v i as vw'.ctn.acc v i';
    };
  varray_implode (from_array vw' (core a));
}

inline_for_extraction noextract
fn varray_view_equiv
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  (a : varray vw)
  (vw' : aview et st { view_equiv vw vw' })
  (#f : perm)
  (#v : erased st)
  requires
    a |-> Frac f v
  returns
    va : varray vw'
  ensures
    va |-> Frac f v **
    pure (core va == core a)
{
  varray_view_equiv_ a vw';
  from_array vw' (core a)
}

ghost
fn varray_split2_
  (#et : Type0) (#st1 #st2 : Type)
  (vw1 : aview et st1)
  (vw2 : aview et st2 { len vw1 = len vw2 })
  (#_ : squash (no_overlap vw1.iview.step.imap.f vw2.iview.step.imap.f))
  (a : varray (sum_aview vw1 vw2))
  (#f : perm)
  (#v : st1 & st2)
  requires
    a |-> Frac f v
  ensures
    (from_array vw1 (core a) |-> Frac f (fst v)) **
    (from_array vw2 (core a) |-> Frac f (snd v))
{
  unfold varray_pts_to a #f v;

  IArray.iarray_split2_ _ _ a;

  IArray.iarray_ext
    (IArray.from_array vw1.iview (IArray.core a))
    (fun i -> (sum_aview vw1 vw2).ctn.acc v (Inl i))
    (vw1.ctn.acc (fst v));
  rewrite each (IArray.from_array vw1.iview (IArray.core a))
            as (from_array vw1 (core a));
  fold varray_pts_to (from_array vw1 (core a)) #f (fst v);

  IArray.iarray_ext
    (IArray.from_array vw2.iview (IArray.core a))
    (fun i -> (sum_aview vw1 vw2).ctn.acc v (Inr i))
    (vw2.ctn.acc (snd v));
  rewrite each (IArray.from_array vw2.iview (IArray.core a))
            as (from_array vw2 (core a));
  fold varray_pts_to (from_array vw2 (core a)) #f (snd v);

  ();
}

inline_for_extraction noextract
fn varray_split2
  (#et : Type0) (#st1 #st2 : Type)
  (vw1 : aview et st1)
  (vw2 : aview et st2 { len vw1 = len vw2 })
  (#_ : squash (no_overlap vw1.iview.step.imap.f vw2.iview.step.imap.f))
  (a : varray (sum_aview vw1 vw2))
  (#f : perm)
  (#v : erased (st1 & st2))
  requires
    a |-> Frac f v
  returns
    a1a2 : varray vw1 & varray vw2
  ensures
    (fst a1a2 |-> Frac f (fst v)) **
    (snd a1a2 |-> Frac f (snd v)) **
    pure (core (fst a1a2) == core a) **
    pure (core (snd a1a2) == core a)
{
  varray_split2_ vw1 vw2 a;
  (from_array vw1 (core a), from_array vw2 (core a));
}

ghost
fn varray_split_n
  (#et : Type0) (#st : Type)
  (#n : pos)
  (vw : natlt n -> aview et st { forall i. len (vw i) = len (vw 0) })
  (#_ : squash (no_overlap_fam n vw))
  (a : varray (sum_aview_fam n vw))
  (#f : perm)
  (#v : natlt n ^->> st)
  requires
    a |-> Frac f v
  ensures
    forall+ (i : natlt n).
      from_array (vw i) (core a) |-> Frac f (v i)
{
  unfold varray_pts_to a #f v;
  IArray.iarray_split_n (fun i -> (vw i).iview) a;
  ghost
  fn aux (i : natlt n)
    requires
      IArray.iarray_pts_to (IArray.from_array (vw i).iview (IArray.core a)) #f
        (fun j -> (sum_aview_fam n vw).ctn.acc v (| i, j |))
    ensures
      varray_pts_to (from_array (vw i) (core a)) #f (v i)
  {
    rewrite each
      IArray.from_array (vw i).iview (IArray.core a)
    as
      (from_array (vw i) (core a));
    IArray.iarray_ext
      (from_array (vw i) (core a))
      (fun j -> (sum_aview_fam n vw).ctn.acc v (| i, j |))
      ((vw i).ctn.acc (v i));
    fold varray_pts_to (from_array (vw i) (core a)) #f (v i);
    ()
  };
  forevery_map _ _ aux;
  ();
}

ghost
fn forevery_join_either'
  (#a #b : Type0)
  (p : a -> slprop)
  (q : b -> slprop)
  requires
    forall+ (x:a). p x
  requires
    forall+ (x:b). q x
  ensures
    forall+ (x:either a b). merge_either p q x
{
  forevery_map p (fun x -> merge_either p q (Inl x))
    fn x { fold merge_either p q (Inl x) };
  forevery_map q (fun x -> merge_either p q (Inr x))
    fn x { fold merge_either p q (Inr x) };
  forevery_join_either (merge_either p q)
}

ghost
fn varray_join2_
  (#et : Type0) (#st1 #st2 : Type)
  (#vw1 : aview et st1)
  (#vw2 : aview et st2 { len vw1 = len vw2 })
  (#_ : squash (no_overlap vw1.iview.step.imap.f vw2.iview.step.imap.f))
  (al : varray vw1)
  (ar : varray vw2)
  (#f : perm)
  (#v1 : st1)
  (#v2 : st2)
  requires pure (core al == core ar)
  requires
    al |-> Frac f v1 **
    ar |-> Frac f v2
  ensures
    (* ARGH AGAIN *)
    from_array (sum_aview vw1 vw2 #()) (core al) |-> Frac f (v1, v2)
{
  varray_pts_to_ref al;
  varray_pts_to_ref ar;
  varray_explode al;
  varray_explode ar;
  forevery_join_either'
    (fun (i: vw1.iview.ait) -> varray_pts_to_cell al #f i (vw1.ctn.acc v1 i))
    (fun (i: vw2.iview.ait) -> varray_pts_to_cell ar #f i (vw2.ctn.acc v2 i));
  forevery_map
    (fun (x: either vw1.iview.ait vw2.iview.ait) ->
      merge_either (fun i -> varray_pts_to_cell al #f i (vw1.ctn.acc v1 i))
        (fun i -> varray_pts_to_cell ar #f i (vw2.ctn.acc v2 i))
        x)
    (fun (i: (sum_aview vw1 vw2).iview.ait) ->
      varray_pts_to_cell (from_array (sum_aview vw1 vw2) (core al)) #f
        i
        ((sum_aview vw1 vw2).ctn.acc (v1, v2) i))
    fn x {
      match x {
        Inl i -> {
          unfold merge_either
            (fun i -> varray_pts_to_cell al #f i (vw1.ctn.acc v1 i))
            (fun i -> varray_pts_to_cell ar #f i (vw2.ctn.acc v2 i))
            (Inl i);
          varray_cell_reindex al i (from_array (sum_aview vw1 vw2) (core al)) x;
        }
        Inr i -> {
          unfold merge_either
            (fun i -> varray_pts_to_cell al #f i (vw1.ctn.acc v1 i))
            (fun i -> varray_pts_to_cell ar #f i (vw2.ctn.acc v2 i))
            (Inr i);
          varray_cell_reindex ar i (from_array (sum_aview vw1 vw2) (core al)) x;
        }
      }
    };
  varray_implode (from_array (sum_aview vw1 vw2) (core al)) #f #(v1, v2);
}

inline_for_extraction noextract
fn varray_join2
  (#et : Type0) (#st1 #st2 : Type)
  (#vw1 : aview et st1)
  (#vw2 : aview et st2 { len vw1 = len vw2 })
  (#_ : squash (no_overlap vw1.iview.step.imap.f vw2.iview.step.imap.f))
  (al : varray vw1)
  (ar : varray vw2)
  (#f : perm)
  (#v1 : erased st1)
  (#v2 : erased st2)
  requires pure (core al == core ar)
  requires
    al |-> Frac f v1 **
    ar |-> Frac f v2
  returns
    a : varray (sum_aview vw1 vw2)
  ensures
    (a |-> Frac f (reveal v1, reveal v2)) **
    pure (core a == core al)
{
  varray_join2_ al ar;
  from_array (sum_aview vw1 vw2) (core al)
}

ghost
fn varray_share_n
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  (a : varray vw)
  (k : pos)
  (#f : perm)
  (#v : st)
  requires
    a |-> Frac f v
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) v
{
  unfold varray_pts_to a #f v;
  IArray.iarray_share_n a k;
  forevery_map
    (fun (i: natlt k) ->
      IArray.iarray_pts_to a #(f /. k) (vw.ctn.acc v))
    (fun (i: natlt k) ->
      varray_pts_to #et a #(f /. k) v)
    fn i {
      fold varray_pts_to a #(f /. k) v;
    }
}

ghost
fn varray_gather_n
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  (a : varray vw)
  (k : pos)
  (#f : perm)
  (#v : st)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) v
  ensures
    a |-> Frac f v
{
  forevery_map
    (fun (i: natlt k) ->
      varray_pts_to #et a #(f /. k) v)
    (fun (i: natlt k) ->
      IArray.iarray_pts_to a #(f /. k) (vw.ctn.acc v))
    fn i {
      unfold varray_pts_to a #(f /. k) v;
    };
  IArray.iarray_gather_n a k;
  fold varray_pts_to a #f v;
}

ghost
fn varray_pts_to_eq
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  (a : varray vw)
  (#f1 f2 : perm)
  (#v1 #v2 : st)
  requires
    a |-> Frac f1 v1 **
    a |-> Frac f2 v2
  ensures
    a |-> Frac f1 v2 **
    a |-> Frac f2 v2
{
  unfold varray_pts_to a #f1 v1;
  unfold varray_pts_to a #f2 v2;
  IArray.iarray_pts_to_eq a #f1 f2;
  fold varray_pts_to a #f1 v2;
  fold varray_pts_to a #f2 v2;
}

inline_for_extraction noextract
fn varray_write_cell
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  {| cw : cview vw |}
  (a : varray vw)
  (ci : cw.sch.cit)
  (v1 : et)
  (#v0 : erased et)
  requires
    Cell a (ci_to_ai vw ci) |-> v0
  ensures
    Cell a (ci_to_ai vw ci) |-> v1
{
  unfold varray_pts_to_cell a (ci_to_ai vw ci) v0;
  IArray.iarray_write_cell a ci v1;
  fold varray_pts_to_cell a (ci_to_ai vw ci) v1;
}

inline_for_extraction noextract
fn varray_write_cell'
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  {| cw : cview vw |}
  (a : varray vw)
  (ai : erased vw.iview.ait)
  (ci : cw.sch.cit)
  (v1 : et)
  (#v0 : erased et)
  requires
    (Cell a (reveal ai) |-> reveal v0) **
    pure (ai == ci_to_ai vw ci)
  ensures
    Cell a (reveal ai) |-> reveal v1
{
  rewrite each reveal ai as (ci_to_ai vw ci);
  varray_write_cell a ci v1;
  rewrite each (ci_to_ai vw ci) as (reveal ai);
}

inline_for_extraction noextract
fn varray_read_cell
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  {| cw : cview vw |}
  (a : varray vw)
  (ci : cw.sch.cit)
  (#f : perm)
  (#v0 : erased et)
  requires
    varray_pts_to_cell a #f (ci_to_ai vw ci) v0
  returns
    v : et
  ensures
    varray_pts_to_cell a #f (ci_to_ai vw ci) v **
    pure (v == v0)
{
  unfold varray_pts_to_cell a #f (ci_to_ai vw ci) v0;
  let res = IArray.iarray_read_cell a ci;
  fold varray_pts_to_cell a #f (ci_to_ai vw ci) v0;
  res
}

inline_for_extraction noextract
fn varray_read_cell'
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  {| cw : cview vw |}
  (a : varray vw)
  (i : cw.sch.cit)
  (ai : erased vw.iview.ait)
  (#f : perm)
  (#v0 : erased et)
  requires
    (Cell a (reveal ai) |-> Frac f v0) **
    pure (ai == ci_to_ai vw i)
  returns
    v : et
  ensures
    (Cell a (reveal ai) |-> Frac f v) **
    pure (v == v0)
{
  rewrite each reveal ai as (ci_to_ai vw i);
  let res = varray_read_cell a i;
  rewrite each (ci_to_ai vw i) as (reveal ai);
  res
}

inline_for_extraction noextract
fn varray_read
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  {| cw : cview vw |}
  (a : varray vw)
  (ci : cw.sch.cit)
  (#f : perm)
  (#v : erased st)
  preserves
    a |-> Frac f v
  returns
    e : et
  ensures
    pure (e == vw.ctn.acc v (ci_to_ai vw ci))
{
  unfold varray_pts_to a #f v;
  let res = IArray.iarray_read a ci;
  fold varray_pts_to a #f v;
  res
}


inline_for_extraction noextract
fn varray_write
  (#et : Type) (#st : Type)
  (#vw : aview et st)
  {| cw : cview vw |}
  (a : varray vw)
  (ci : cw.sch.cit)
  (e : et)
  (#v0 : erased st)
  requires
    a |-> v0
  ensures
    a |-> vw.ctn.upd v0 (ci_to_ai vw ci) e
{
  unfold varray_pts_to a v0;
  IArray.iarray_write a ci e;
  Container.oplus_lemma (reveal v0) (ci_to_ai vw ci) e;
  IArray.iarray_ext a
    _
    (vw.ctn.acc (vw.ctn.upd v0 (ci_to_ai vw ci) e));
  fold varray_pts_to a (vw.ctn.upd v0 (ci_to_ai vw ci) e);
  ()
}

inline_for_extraction noextract
fn varray_from_array
  (#et:Type0) {| sized et |} (#st : Type0)
  (#vw : aview et st { is_full_view vw })
  (clen : sz { SZ.v clen == vw.iview.len})
  (va : varray vw)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == len vw})
  (#v : erased st)
  preserves
    a |-> s **
    cpu
  requires
    on gpu_loc (va |-> v)
  ensures
    pure (Pulse.Lib.Vec.length a == len vw) **
    on gpu_loc (va |-> from_seq vw s) //TODO: consider rebinding
{
  // let len = cw.clen;
  Pulse.Lib.Vec.pts_to_len a;
  map_loc gpu_loc (fun () -> varray_concr va);
  B.gpu_memcpy_host_to_device (core va) a clen;
  map_loc gpu_loc (fun () -> varray_abs_alt' vw _ (core va));
  rewrite each from_array vw (core va) as va;
  ();
}

inline_for_extraction noextract
fn varray_to_array
  (#et:Type0) {| sized et |} (#st : Type0)
  (#vw : aview et st { is_full_view vw })
  (clen : sz { SZ.v clen == vw.iview.len})
  (a : vec et)
  (va : varray vw)
  (#s : erased (seq et){Seq.length s == len vw})
  (#v : erased st)
  preserves
    on gpu_loc (va |-> v) **
    cpu
  requires
    a |-> s
  ensures
    pure (Pulse.Lib.Vec.length a == len vw) **
    (a |-> to_seq vw v)
{
  Pulse.Lib.Vec.pts_to_len a;
  map_loc gpu_loc (fun () -> varray_concr va);
  B.gpu_memcpy_device_to_host a (core va) clen;
  map_loc gpu_loc (fun () -> varray_abs' vw (core va));
  rewrite each
    (from_array vw (core va) |-> from_seq vw (to_seq vw v))
  as
    (va |-> v);
}
