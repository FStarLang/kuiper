module Kuiper.VArray
#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.Injection
module Enum = Kuiper.Enumerable
module B = Kuiper.Array (* base *)
module T = FStar.Tactics.V2
module SZ = FStar.SizeT
module Trade = Pulse.Lib.Trade
module IView = Kuiper.IView
module IArray = Kuiper.IArray

noeq
inline_for_extraction
type varray
  (#et:Type0) (#len : erased nat) (#st : Type0)
  (vw : aview et len st)
= | VA of IArray.iarray et vw.iview

inline_for_extraction noextract
let from_array
  (#a : Type0) (#len : erased nat) (#st : Type0)
  (vw : aview a len st)
  (arr : gpu_array a len)
  : varray vw
  = VA (IArray.from_array vw.iview arr)

let core (VA a) = IArray.core a

let lem_from_array_core
  (#a : Type0)
  (#len : erased nat)
  (#st : Type0) (#vw : aview a len st)
  (arr : varray vw)
  : Lemma (ensures from_array vw (core arr) == arr)
          [SMTPat (core arr)]
  = ()

let lem_core_from_array
  (#a : Type0)
  (#len : erased nat)
  (#st : Type0) (#vw : aview a len st)
  (p : gpu_array a len)
  : Lemma (ensures core (from_array vw p) == p)
          [SMTPat (from_array vw p)]
  = ()

let varray_pts_to_cell
  (#et:Type0)
  (#len : erased nat) (#st:Type0)
  (#vw : aview et len st)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.iview.ait)
  (v : et)
  : slprop
  = Cell (VA?._0 a) i |-> Frac f v

let varray_pts_to
  (#et:Type0) (#len : erased nat) (#st:_) (#vw : aview et len st)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : st)
  : slprop
  =
    (VA?._0 a) |-> Frac f (fun i -> reveal (vw.igm.acc v i))

ghost
fn varray_pts_to_ref
  (#t:Type0)
  (#len : erased nat)
  (#st:Type0)
  (#vw : aview t len st)
  (a : varray vw)
  (#f : perm)
  (#v : erased st)
  preserves
    a |-> Frac f v
  ensures
    pure (SZ.fits len)
{
  unfold varray_pts_to a #f v;
  IArray.iarray_pts_to_ref (VA?._0 a);
  fold varray_pts_to a #f v;
  ()
}

ghost
fn varray_explode
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  (a : varray vw)
  (#f : perm)
  (#v : st)
  requires
    a |-> Frac f v
  ensures
    forall+ (i : vw.iview.ait).
      Cell a i |-> Frac f (vw.igm.acc v i)
{
  unfold varray_pts_to a #f v;
  IArray.iarray_explode (VA?._0 a);
  ghost
  fn aux (i : vw.iview.ait)
    requires
      Cell (VA?._0 a) i |-> Frac f (vw.igm.acc v i)
    ensures
      Cell a i |-> Frac f (vw.igm.acc v i)
  {
    fold varray_pts_to_cell a #f i (vw.igm.acc v i);
  };
  forevery_map _ _ aux;
  ()
}

ghost
fn varray_implode
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  (a : varray vw)
  (#f : perm)
  (#v : st)
  requires
    pure (SZ.fits len)
  requires
    forall+ (i : vw.iview.ait).
      Cell a i |-> Frac f (vw.igm.acc v i)
  ensures
    a |-> Frac f v
{
  ghost
  fn aux (i : vw.iview.ait)
    requires
      Cell a i |-> Frac f (vw.igm.acc v i)
    ensures
      Cell (VA?._0 a) i |-> Frac f (vw.igm.acc v i)
  {
    unfold varray_pts_to_cell a #f i (vw.igm.acc v i);
  };
  forevery_map _ _ aux;
  IArray.iarray_implode (VA?._0 a);
  fold varray_pts_to a #f v;
}

(* Begin viewing something abstractly, with the trivial view. The spec
type are sequences. *)
ghost
fn varray_begin_
  (#et : Type) (#len : erased nat)
  (a : gpu_array et len)
  (#f : perm)
  (#v : lseq et len)
  requires
    a |-> Frac f v
  ensures
    from_array (raw_view #et #len) a |-> Frac f v
{
  IArray.iarray_begin_ a;
  IArray.iarray_ext (IArray.from_array (IView.raw_view #len) a)
    (IArray.g_seq_acc v)
    (fun i -> (raw_view #et #len).igm.acc v i);
  fold varray_pts_to (from_array (raw_view #et #len) a ) #f v;
}

inline_for_extraction noextract
fn varray_begin
  (#et : Type) (#len : erased nat)
  (a : gpu_array et len)
  (#f : perm)
  (#v : erased (lseq et len))
  requires
    a |-> Frac f v
  returns
    va : varray (raw_view #et #len)
  ensures
    (va |-> Frac f v) **
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
  IArray.iarray_end_ (VA?._0 a);
  rewrite each IArray.core #et #len #(IView.raw_view #len) a._0 as core a;
  with vv.
    assert B.gpu_pts_to_slice (core a) #f 0 len vv;
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
    a' : gpu_array et len
  ensures
    a' |-> Frac f v
{
  varray_end_ a;
  (core a);
}

ghost
fn varray_abs
  (#et : Type0) (#len : erased nat) (#st : Type0)
  (vw : aview et len st { is_full_view vw })
  (a : gpu_array et len)
  (#f : perm)
  (#v : st)
  requires
    a |-> Frac f (to_seq vw v)
  ensures
    from_array vw a |-> Frac f v
{
  admit();
}

ghost
fn varray_abs'
  (#et : Type0) (#len : erased nat) (#st : Type0)
  (vw : aview et len st { is_full_view vw })
  (a : gpu_array et len)
  (#f : perm)
  (#v : lseq et len)
  requires
    a |-> Frac f v
  ensures
    from_array vw a |-> Frac f (from_seq vw v)
{
  admit();
}

ghost
fn varray_concr
  (#et : Type0) (#len : erased nat) (#st : Type0)
  (#vw : aview et len st { is_full_view vw })
  (a : varray vw)
  (#f : perm)
  (#v : erased st)
  requires
    a |-> Frac f v
  ensures
    core a |-> Frac f (to_seq vw v)
{
  admit();
}


inline_for_extraction noextract
fn varray_alloc0
  (#et : Type0) {| sized et |} (len : sz) (#st : Type0)
  (vw : aview et len st { is_full_view vw })
  preserves
    cpu
  requires
    pure (SZ.fits len)
  returns
    a : varray vw
  ensures
    exists* v. a |-> v
{
  let a = B.gpu_array_alloc #et len;
  varray_abs' vw a;
  (from_array vw a)
}

inline_for_extraction noextract
fn varray_free
  (#et : Type0) (#len : erased nat) (#st : Type0)
  (#vw : aview et len st { is_full_view vw })
  (a : varray vw)
  (#v : erased st)
  preserves
    cpu
  requires
    a |-> v
  ensures emp
{
  varray_concr a;
  B.gpu_array_free (core a);
}


(* Note how the spec type does not change at all. The mapping
is hidden in the view. *)
ghost
fn varray_reindex_
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  (#ait' : Type)
  {| Enumerable.enumerable ait' |}
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
  IArray.iarray_reindex_ bij a._0;
  rewrite each
    IArray.from_array #et #len (IView.reindex_view #len vw.iview #ait' bij) (IArray.core a._0)
  as
    (from_array (reindex_view vw bij) (core a))._0;
  IArray.iarray_ext (from_array (reindex_view vw bij) (core a))._0
    _
    (fun i -> (reindex_view vw bij).igm.acc v i);
  fold varray_pts_to (from_array (reindex_view vw bij) (core a)) #f v;
  ()
}

inline_for_extraction noextract
fn varray_reindex
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  (#ait' : Type)
  {| Enumerable.enumerable ait' |}
  (bij : vw.iview.ait =~ ait')
  (a : varray vw)
  (#f : perm)
  (#v : erased st)
  requires
    a |-> Frac f v
  returns
    va : varray (reindex_view vw bij)
  ensures
    (va |-> Frac f v) **
    pure (core va == core a)
{
  varray_reindex_ bij a;
  from_array (reindex_view vw bij) (core a)
}

ghost
fn varray_review_
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
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
  IArray.iarray_ext a._0
    (fun i -> vw.igm.acc v i)
    (fun i -> (review_view vw bij).igm.acc (bij.ff v) i);

  rewrite IArray.iarray_pts_to #et #len #vw.iview a._0 #f (fun i -> (review_view vw bij).igm.acc (bij.ff v) i)
       as IArray.iarray_pts_to #et #len #vw.iview
            (from_array (review_view vw bij) (core a))._0
            #f
            (fun i -> (review_view vw bij).igm.acc (bij.ff v) i);
  assert pure (vw.iview == (review_view vw bij).iview);
  rewrite IArray.iarray_pts_to #et #len #vw.iview
            (from_array (review_view vw bij) (core a))._0
            #f
            (fun i -> (review_view vw bij).igm.acc (bij.ff v) i)
       as IArray.iarray_pts_to #et #len #(review_view vw bij).iview
            (from_array (review_view vw bij) (core a))._0
            #f
            (fun i -> (review_view vw bij).igm.acc (bij.ff v) i)
       by slprop_equiv_norm (); // ugly
  fold varray_pts_to (from_array (review_view vw bij) (core a)) #f (bij.ff v);
  ()
}

inline_for_extraction noextract
fn varray_review
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
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

ghost
fn varray_view_equiv_
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  (a : varray vw)
  (vw' : aview et len st)
  (#f : perm)
  (#v : erased st)
  requires
    a |-> Frac f v
  ensures
    from_array vw' (core a) |-> Frac f v
{
  // need more requirements of course!!!
  admit();
}

inline_for_extraction noextract
fn varray_view_equiv
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  (a : varray vw)
  (vw' : aview et len st)
  (#f : perm)
  (#v : erased st)
  requires
    a |-> Frac f v
  returns
    va : varray vw'
  ensures
    (va |-> Frac f v) **
    pure (core va == core a)
{
  varray_view_equiv_ a vw';
  from_array vw' (core a)
}


ghost
fn varray_split2_
  (#et : Type0) (#len : nat) (#st1 #st2 : Type)
  (vw1 : aview et len st1)
  (vw2 : aview et len st2)
  (#_ : squash (no_overlap vw1.iview.imap.f vw2.iview.imap.f))
  (a : varray (sum_aview vw1 vw2 #())) /// argh!!!! affects typeclass resolution!!!!
  (#f : perm)
  (#v : st1 & st2)
  requires
    a |-> Frac f v
  ensures
    (from_array vw1 (core a) |-> Frac f (fst v)) **
    (from_array vw2 (core a) |-> Frac f (snd v))
{
  unfold varray_pts_to a #f v;

  IArray.iarray_split2_ _ _ a._0;

  IArray.iarray_ext
    (IArray.from_array vw1.iview (IArray.core a._0))
    (fun i -> (sum_aview vw1 vw2).igm.acc v (Inl i))
    (fun i -> vw1.igm.acc (fst v) i);
  rewrite each (IArray.from_array vw1.iview (IArray.core a._0))
            as (from_array vw1 (core a))._0;
  fold varray_pts_to (from_array vw1 (core a)) #f (fst v);

  IArray.iarray_ext
    (IArray.from_array vw2.iview (IArray.core a._0))
    (fun i -> (sum_aview vw1 vw2).igm.acc v (Inr i))
    (fun i -> vw2.igm.acc (snd v) i);
  rewrite each (IArray.from_array vw2.iview (IArray.core a._0))
            as (from_array vw2 (core a))._0;
  fold varray_pts_to (from_array vw2 (core a)) #f (snd v);

  ();
}

inline_for_extraction noextract
fn varray_split2
  (#et : Type0) (#len : nat) (#st1 #st2 : Type)
  (vw1 : aview et len st1)
  (vw2 : aview et len st2)
  (#_ : squash (no_overlap vw1.iview.imap.f vw2.iview.imap.f))
  (a : varray (sum_aview vw1 vw2 #())) /// argh!!!! affects typeclass resolution!!!!
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
fn varray_join2_
  (#et : Type0) (#len : nat) (#st1 #st2 : Type)
  (#vw1 : aview et len st1)
  (#vw2 : aview et len st2)
  (#_ : squash (no_overlap vw1.iview.imap.f vw2.iview.imap.f))
  (al : varray vw1)
  (ar : varray vw2)
  (#f : perm)
  (#v1 : st1)
  (#v2 : st2)
  requires pure (core al == core ar)
  requires
    (al |-> Frac f v1) **
    (ar |-> Frac f v2)
  ensures
    (* ARGH AGAIN *)
    from_array (sum_aview vw1 vw2 #()) (core al) |-> Frac f (v1, v2)
{
  admit();
}

inline_for_extraction noextract
fn varray_join2
  (#et : Type0) (#len : nat) (#st1 #st2 : Type)
  (#vw1 : aview et len st1)
  (#vw2 : aview et len st2)
  (#_ : squash (no_overlap vw1.iview.imap.f vw2.iview.imap.f))
  (al : varray vw1)
  (ar : varray vw2)
  (#f : perm)
  (#v1 : erased st1)
  (#v2 : erased st2)
  requires pure (core al == core ar)
  requires
    (al |-> Frac f v1) **
    (ar |-> Frac f v2)
  returns
    a : varray (sum_aview vw1 vw2 #())
  ensures
    (a |-> Frac f (reveal v1, reveal v2)) **
    pure (core a == core al)
{
  varray_join2_ al ar;
  from_array (sum_aview vw1 vw2 #()) (core al)
}

// TODO: remove?
ghost
fn varray_share_n
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  (#[T.exact (`0)] uid: int)
  (a : varray vw)
  (k : pos)
  (#f : perm)
  (#v : st)
  requires
    a |-> Frac f v
  ensures
    bigstar #uid 0 k (fun _ -> a |-> Frac (f /. k) v)
{
  unfold varray_pts_to a #f v;
  IArray.iarray_share_n #_ #_ #_ #uid (VA?._0 a) k;
  ghost
  fn aux (i : nat)
    requires IArray.iarray_pts_to a._0 #(f /. k) (fun i -> vw.igm.acc v i)
    ensures  varray_pts_to #et #len a #(f /. k) v
  {
    fold varray_pts_to a #(f /. k) v;
  };
  bigstar_map #uid #uid aux;
  ();
}

// TODO: remove?
ghost
fn varray_gather_n
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  (#[T.exact (`0)] uid: int)
  (a : varray vw)
  (k : pos)
  (#f : perm)
  (#v : st)
  requires
    bigstar #uid 0 k (fun _ -> a |-> Frac (f /. k) v)
  ensures
    a |-> Frac f v
{
  ghost
  fn aux (i : nat)
    requires varray_pts_to #et #len a #(f /. k) v
    ensures  IArray.iarray_pts_to a._0 #(f /. k) (fun i -> vw.igm.acc v i)
  {
    unfold varray_pts_to a #(f /. k) v;
  };
  bigstar_map #uid #uid aux;
  IArray.iarray_gather_n #_ #_ #_ #uid (VA?._0 a) k;
  fold varray_pts_to a #f v;
}

inline_for_extraction noextract
fn varray_write_cell
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  {| cw : cview vw |}
  (a : varray vw)
  (ci : cw.cit)
  (v1 : et)
  (#v0 : erased et)
  preserves gpu
  requires
    Cell a (ci_to_ai vw ci) |-> v0
  ensures
    Cell a (ci_to_ai vw ci) |-> v1
{
  unfold varray_pts_to_cell a (ci_to_ai vw ci) v0;
  IArray.iarray_write_cell (VA?._0 a) ci v1;
  fold varray_pts_to_cell a (ci_to_ai vw ci) v1;
}

inline_for_extraction noextract
fn varray_write_cell'
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  {| cw : cview vw |}
  (a : varray vw)
  (ai : erased vw.iview.ait)
  (ci : cw.cit)
  (v1 : et)
  (#v0 : erased et)
  preserves
    gpu
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
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  {| cw : cview vw |}
  (a : varray vw)
  (ci : cw.cit)
  (#f : perm)
  (#v0 : erased et)
  preserves
    gpu
  requires
    varray_pts_to_cell a #f (ci_to_ai vw ci) v0
  returns
    v : et
  ensures
    varray_pts_to_cell a #f (ci_to_ai vw ci) v **
    pure (v == v0)
{
  unfold varray_pts_to_cell a #f (ci_to_ai vw ci) v0;
  let res = IArray.iarray_read_cell (VA?._0 a) ci;
  fold varray_pts_to_cell a #f (ci_to_ai vw ci) v0;
  res
}

inline_for_extraction noextract
fn varray_read_cell'
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (ai : erased vw.iview.ait)
  (#f : perm)
  (#v0 : erased et)
  preserves
    gpu
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
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  {| cw : cview vw |}
  (a : varray vw)
  (ci : cw.cit)
  (#f : perm)
  (#v : erased st)
  preserves
    gpu **
    (a |-> Frac f v)
  returns
    e : et
  ensures
    pure (e == vw.igm.acc v (ci_to_ai vw ci))
{
  unfold varray_pts_to a #f v;
  let res = IArray.iarray_read (VA?._0 a) ci;
  fold varray_pts_to a #f v;
  res
}

inline_for_extraction noextract
fn varray_write
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  {| cw : cview vw |}
  (a : varray vw)
  (ci : cw.cit)
  (e : et)
  (#v0 : erased st)
  preserves
    gpu
  requires
    a |-> v0
  ensures
    a |-> vw.igm.upd v0 (ci_to_ai vw ci) e
{
  unfold varray_pts_to a v0;
  IArray.iarray_write a._0 ci e;
  vw.igm.l2 (ci_to_ai vw ci) v0 e;
  IArray.iarray_ext a._0
    _
    (fun i -> vw.igm.acc (vw.igm.upd v0 (ci_to_ai vw ci) e) i);
  fold varray_pts_to a (vw.igm.upd v0 (ci_to_ai vw ci) e);
  ()
}


inline_for_extraction noextract
fn varray_from_array
  (#et:Type0) {| sized et |} (#len : SZ.t) (#st : Type0)
  (#vw : aview et len st { is_full_view vw })
  (va : varray vw)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == len})
  (#v : erased st)
  preserves
    (a |-> s) **
    cpu
  requires
    (va |-> v)
  ensures
    pure (SZ.fits len /\ Pulse.Lib.Vec.length a == len) **
    (va |-> from_seq vw s)
{
  Pulse.Lib.Vec.pts_to_len a;
  varray_concr va;
  B.gpu_memcpy_host_to_device (core va) a len;
  varray_abs' vw (core va);
  rewrite each from_array vw (core va) as va;
  ();
}

inline_for_extraction noextract
fn varray_to_array
  (#et:Type0) {| sized et |} (#len : SZ.t) (#st : Type0)
  (#vw : aview et len st { is_full_view vw })
  (a : vec et)
  (va : varray vw)
  (#s : erased (seq et){Seq.length s == len})
  (#v : erased st)
  preserves
    (va |-> v) **
    cpu
  requires
    (a |-> s)
  ensures
    pure (SZ.fits len /\ Pulse.Lib.Vec.length a == len) **
    (a |-> to_seq vw v)
{
  Pulse.Lib.Vec.pts_to_len a;
  varray_concr va;
  B.gpu_memcpy_device_to_host a (core va) len;
  varray_abs' vw (core va);
  rewrite each from_array vw (core va) as va;
  rewrite each from_seq vw (to_seq vw v) as v;
  ();
}
