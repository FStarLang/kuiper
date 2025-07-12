module Kuiper.Matrix4
#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.GhostMap
open Kuiper.EMatrix4
module A = Kuiper.ArrayView
module T = FStar.Tactics.V2
open FStar.SizeT { div as (/^), (%^), (+^), (-^), ( *^ )  }

inline_for_extraction noextract
type cit
  (#mrows #mcols #brows #bcols : erased nat)
  (l : mlayout4 mrows mcols brows bcols)
  : Type
  = szlt mrows & szlt mcols &
    szlt brows & szlt bcols

inline_for_extraction noextract
let cview_from_clayout_ff
  (et : Type)
  (#mrows #brows #mcols #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (c : clayout4 l)
  : cit l -> szlt (mrows * mcols * brows * bcols)
  = fun (bi, bj, i, j) ->
      c.parent.c_to
        (s_undivmod c.c_brows (bi, i))
        (s_undivmod c.c_bcols (bj, j))

inline_for_extraction noextract
let cview_from_clayout_gg
  (et : Type)
  (#mrows #brows #mcols #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (c : clayout4 l)
  : szlt (mrows * mcols * brows * bcols) -> cit l
  = fun x ->
      [@@inline_let] let i = c.parent.c_from1 x in
      [@@inline_let] let j = c.parent.c_from2 x in
      [@@inline_let] let bi, si = s_divmod c.c_brows i in
      [@@inline_let] let bj, sj = s_divmod c.c_bcols j in
      (bi, bj, si, sj)

let cview_from_clayout_gg_ff
  (et : Type)
  (#mrows #brows #mcols #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (c : clayout4 l)
  (i4 : cit l)
  : squash (cview_from_clayout_gg et c (cview_from_clayout_ff et c i4) == i4)
= calc (==) {
    cview_from_clayout_gg et c (cview_from_clayout_ff et c i4);
    == {}
    (let i = c.parent.c_from1 (cview_from_clayout_ff et c i4) in
     let j = c.parent.c_from2 (cview_from_clayout_ff et c i4) in
     let bi, si = s_divmod c.c_brows i in
     let bj, sj = s_divmod c.c_bcols j in
     (bi, bj, si, sj));
  }

inline_for_extraction noextract
instance cview_from_clayout4
  (et : Type)
  (#mrows #mcols : erased nat)
  (#brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (c : clayout4 l)
  : A.cview (aview_from_mlayout et l) =
{
  cit = cit l;
  cibij = {
    ff = cview_from_clayout_ff et c;
    gg = cview_from_clayout_gg et c;
    ff_gg = ez;
    gg_ff = cview_from_clayout_gg_ff et c;
  }
}

inline_for_extraction noextract
let gpu_matrix
  (et:Type0)
  (#mrows #mcols #brows #bcols : nat)
  (l : mlayout4 mrows mcols brows bcols)
  : Type0
  = A.varray (aview_from_mlayout et l)

let from_array l p = A.from_array (aview_from_mlayout _ l) p
let core g = A.core g

let lem_core_from_array
  (#et : Type)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (g : gpu_matrix et l)
  : Lemma (ensures from_array l (core g) == g)
          [SMTPat (core g)]
  = ()

let lem_from_array_core
  (#et : Type)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (p : gpu_array et (mlayout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = admit()

let gpu_matrix_pts_to
  (#et:Type) (#mrows #mcols #brows #bcols : nat)
  (#l : mlayout4 mrows mcols brows bcols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  (em : _)
  : slprop
  = A.varray_pts_to gm #f em

ghost
fn gpu_matrix_pts_to_ref
  (#et:Type)
  (#mrows #mcols #brows #bcols : nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (g : gpu_matrix et l)
  (#f : perm)
  (#em : ematrix4 et mrows mcols brows bcols)
  preserves
    gpu_matrix_pts_to g #f em
  ensures
    pure (SZ.fits (mlayout_size l))
{
  unfold gpu_matrix_pts_to g #f em;
  A.varray_pts_to_ref g;
  fold gpu_matrix_pts_to g #f em;
}

ghost
fn gpu_matrix_concr
  (#et:Type)
  (#mrows #mcols #brows #bcols : nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (g : gpu_matrix et l)
  (#em : ematrix4 et mrows mcols brows bcols)
  (#f : perm)
  requires
    g |-> Frac f em
  ensures
    core g |-> Frac f (to_seq l em)
{
  unfold gpu_matrix_pts_to g #f em;
  let a' = A.varray_concr g;
  assert (pure (Seq.equal (to_seq l em) (A.to_seq (aview_from_mlayout et l) em)));
  a'
}

ghost
fn gpu_matrix_abs
  (#et:Type)
  (#mrows #mcols #brows #bcols : nat)
  (l : mlayout4 mrows mcols brows bcols)
  (p : gpu_array et (mlayout_size l))
  (#f : perm)
  (#em : ematrix4 et mrows mcols brows bcols)
  requires
    p |-> Frac f (to_seq l em)
  ensures
    from_array l p |-> Frac f em
{
  assert (pure (Seq.equal (to_seq l em) (A.to_seq (aview_from_mlayout et l) em)));
  (* FIXME: need to provide implicits very precisely. *)
  rewrite each to_seq #et #(mrows `op_Multiply` brows) #(mcols `op_Multiply` bcols) l em as A.to_seq (aview_from_mlayout et l) em;
  A.varray_abs (aview_from_mlayout et l) p;
  fold gpu_matrix_pts_to (from_array l p) #f em;
}

ghost
fn gpu_matrix_abs'
  (#et:Type)
  (#mrows #mcols #brows #bcols : nat)
  (l : mlayout4 mrows mcols brows bcols)
  (p : gpu_array et (mlayout_size l))
  (#f : perm)
  (#s : lseq et (mlayout_size l))
  requires
    p |-> Frac f s
  ensures
    from_array l p |-> Frac f (from_seq l s)

{
  rewrite each s as to_seq l (from_seq l s);
  gpu_matrix_abs l p;
}

inline_for_extraction noextract
fn gpu_matrix_alloc0
  (#et:Type) {| sized et |}
  (mrows mcols brows bcols : szp)
  (l : mlayout4 mrows mcols brows bcols)
  preserves
    cpu
  requires
    pure (SZ.fits (mlayout_size l))
  returns
    gm : gpu_matrix et l
  ensures
    exists* em. gm |-> em
{
  open FStar.SizeT;
  let gm = A.varray_alloc0 (mrows *^ brows *^ mcols *^ bcols) (aview_from_mlayout et l);
  with s. assert (A.varray_pts_to gm #1.0R s);
  fold gpu_matrix_pts_to gm s;
  gm;
}

inline_for_extraction noextract
fn gpu_matrix_free
  (#et:Type)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (gm : gpu_matrix et l)
  (#em : _)
  preserves
    cpu
  requires
    gm |-> em
  ensures emp
{
  unfold gpu_matrix_pts_to gm em;
  A.varray_free gm;
}

ghost
fn gpu_matrix_share_n
  (#et:Type0)
  (#[T.exact (`0)]uid: int)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  (#em : _)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    bigstar #uid 0 k (fun _ -> gpu_matrix_pts_to gm #(f /. k) em)
{
  unfold gpu_matrix_pts_to gm #f em;
  A.varray_share_n gm k;
  ghost
  fn aux (i:natlt k)
    requires A.varray_pts_to gm #(f /. k) em
    ensures  gpu_matrix_pts_to gm #(f /. k) em
  {
    fold gpu_matrix_pts_to gm #(f /. k) em;
  };
  bigstar_map #0 #uid #0 #k aux;
}

ghost
fn gpu_matrix_gather_n
  (#et:Type0)
  (#uid: int)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  (#em : _)
  requires
    bigstar #uid 0 k (fun _ -> gpu_matrix_pts_to gm #(f /. k) em)
  ensures
    gpu_matrix_pts_to gm #f em
{
  ghost
  fn aux (i:natlt k)
    requires gpu_matrix_pts_to gm #(f /. k) em
    ensures  A.varray_pts_to gm #(f /. k) em
  {
    unfold gpu_matrix_pts_to gm #(f /. k) em;
  };
  bigstar_map #uid #0 #0 #k aux;
  A.varray_gather_n gm k;
  fold gpu_matrix_pts_to gm #f em;
}

inline_for_extraction noextract
fn gpu_matrix_read
  (#et:Type0)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols) {| cl : clayout4 l |}
  (gm : gpu_matrix et l)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i : szlt brows)
  (j : szlt bcols)
  (#f : perm)
  (#em : ematrix4 et mrows mcols brows bcols)
  requires
    gpu **
    gpu_matrix_pts_to gm #f em
  returns v : et
  ensures
    gpu **
    gpu_matrix_pts_to gm #f em **
    pure (v == macc em bi bj i j)
{
  unfold gpu_matrix_pts_to gm #f em;
  let r = A.varray_read gm (bi, bj, i, j);
  fold gpu_matrix_pts_to gm #f em;
  r
}

inline_for_extraction noextract
fn gpu_matrix_write
  (#et:Type0)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols) {| cl : clayout4 l |}
  (gm : gpu_matrix et l)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i : szlt brows)
  (j : szlt bcols)
  (v : et)
  (#em : ematrix4 et mrows mcols brows bcols)
  requires
    gpu **
    gpu_matrix_pts_to gm em
  ensures
    gpu **
    gpu_matrix_pts_to gm (mupd em bi bj i j v)
{
  unfold gpu_matrix_pts_to gm em;
  let cit = (bi, bj, i, j);
  A.varray_write gm cit v;
  let m' = mupd em bi bj i j v;
  assert (pure (
    m'
    `Kuiper.EMatrix4.equal`
    mupd em bi bj i j v));
  assert (pure (A.cit_to_it (aview_from_mlayout et l) #(cview_from_clayout4 et cl) (bi, bj, i, j) ==
    Mktuple2 #(natlt (mrows * brows)) #(natlt (mcols * bcols))
      (SZ.v bi * brows + SZ.v i)
      (SZ.v bj * bcols + SZ.v j))
  );
  assume (pure (
    mupd em bi bj i j v
    `Kuiper.EMatrix4.equal`
    (aview_from_mlayout et l).igm.upd
      em (Mktuple2 #(natlt (mrows * brows)) #(natlt (mcols * bcols))
      (SZ.v bi * brows + SZ.v i)
      (SZ.v bj * bcols + SZ.v j)) v));
  fold gpu_matrix_pts_to gm (mupd em bi bj i j v);
}

let gpu_matrix_pts_to_cell
  (#et:Type0)
  (#mrows #mcols #brows #bcols : nat)
  (#l : mlayout4 mrows mcols brows bcols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] bi : natlt mrows)
  ([@@@mkey] bj : natlt mcols)
  ([@@@mkey] i : natlt brows)
  ([@@@mkey] j : natlt bcols)
  (v : et)
  : slprop
  = A.varray_pts_to_cell gm #f
       (undivmod brows (bi, i),
        undivmod bcols (bj, j)) v

#push-options "--z3rlimit 20" // flaky
inline_for_extraction noextract
fn gpu_matrix_read_cell
  (#et:Type0)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols) {| cl : clayout4 l |}
  (gm : gpu_matrix et l)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i  : szlt brows)
  (j  : szlt bcols)
  (#f : perm)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm #f bi bj i j v0
  returns v : et
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm #f bi bj i j v **
    pure (v == v0)
{
  unfold gpu_matrix_pts_to_cell gm #f bi bj i j v0;
  (* very awkward *)
  with i_low v_low. assert (A.varray_pts_to_cell gm #f i_low v_low);
  rewrite each i_low as
       A.cit_to_it #_ #_ #_
        (aview_from_mlayout et l)
        #(cview_from_clayout4 et cl )
        (bi, bj, i, j);
  assert (pure (brows > 0));
  assert (pure (bcols > 0));
  let v = A.varray_read_cell gm (bi, bj, i, j);
  with i1 v1.
    assert (A.varray_pts_to_cell gm #f i1 v1);
  rewrite A.varray_pts_to_cell gm #f i1 v1 as
    gpu_matrix_pts_to_cell gm #f bi bj i j v0;
  v;
}
#pop-options

inline_for_extraction noextract
fn gpu_matrix_write_cell
  (#et:Type0)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols) {| clayout4 l |}
  (gm : gpu_matrix et l)
  (bi : szlt mrows)
  (bj : szlt mcols)
  (i  : szlt brows)
  (j  : szlt bcols)
  (v1 : et)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm bi bj i j v0
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm bi bj i j v1
{
  // let ci : cit l = (bi, bj, i, j);
  // ^ having this here worsens code generation by introducing
  // more intermediate variables. Why? Every Pulse let is supposed
  // to be marked inline.
  unfold gpu_matrix_pts_to_cell gm bi bj i j v0;
  with i_low v_low.
    assert (A.varray_pts_to_cell gm #1.0R i_low v_low);
  rewrite
    each i_low
      as A.cit_to_it (aview_from_mlayout et l) (bi, bj, i, j);
  A.varray_write_cell gm (bi, bj, i, j) v1;
  with i1 lv1.
    assert (A.varray_pts_to_cell gm i1 lv1);
  rewrite A.varray_pts_to_cell gm i1 lv1 as
    gpu_matrix_pts_to_cell gm bi bj i j v1;
}

ghost
fn gpu_matrix_explode
  (#et:Type0)
  (#mrows #mcols #brows #bcols : nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : _)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    forall+ bi bj i j.
      gpu_matrix_pts_to_cell gm #f bi bj i j (macc em bi bj i j)
{
  unfold gpu_matrix_pts_to gm #f em;
  A.varray_explode gm;
  (* Change the type... convince pulse. *)
  forevery_rw_type
    (aview_from_mlayout et l).it
    (natlt (mrows * brows) & natlt (mcols * bcols))
    (fun rc ->
      A.varray_pts_to_cell gm #f rc ((aview_from_mlayout et l).igm.acc em rc));
  forevery_ext #(natlt (mrows * brows) & natlt (mcols * bcols))
    (fun rc ->
      A.varray_pts_to_cell gm #f rc ((aview_from_mlayout et l).igm.acc em rc))
    (fun rc ->
      A.varray_pts_to_cell gm #f (rc._1, rc._2) (EMatrix.macc em rc._1 rc._2));
  forevery_unflatten #(natlt (mrows * brows)) #_ #(natlt (mcols * bcols))
    (fun r c ->
      A.varray_pts_to_cell gm #f (r, c) (EMatrix.macc em r c));
  forevery_factor (mrows * brows) mrows brows _;
  (* tedious... *)
  admit();
}

ghost
fn gpu_matrix_implode
  (#et:Type0)
  (#mrows #mcols #brows #bcols : nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : _)
  requires
    forall+ bi bj i j.
      gpu_matrix_pts_to_cell gm #f bi bj i j (macc em bi bj i j)
  ensures
    gpu_matrix_pts_to gm #f em
{
  (* same old. *)
  admit();
}

inline_for_extraction noextract
fn gpu_matrix_from_array
  (#et:Type0) {| sized et |}
  (#mrows #mcols #brows #bcols : SZ.t)
  (#l : mlayout4 mrows mcols brows bcols)
  (gm : gpu_matrix et l)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == mlayout_size l})
  (#em : _)
  preserves
    (a |-> s) **
    cpu
  requires
    (* silly, but this shows that the multiplication below does not overflow.
    If we had a mul_underspec, we would not need this, I think. *)
    pure (mlayout_size l > 0) **
    (gm |-> em)
  ensures
    pure (SZ.fits (mlayout_size l) /\ Pulse.Lib.Vec.length a == (mlayout_size l)) **
    (gm |-> from_seq l s)
{
  Pulse.Lib.Vec.pts_to_len a;
  assert (pure (SZ.fits (mlayout_size l)));
  unfold gpu_matrix_pts_to gm #1.0R em;
  let sz = (mrows *^ brows) *^ (mcols *^ bcols);
  A.varray_from_array #_ #_ #sz gm a;
  from_seq_rel l s;
  fold gpu_matrix_pts_to gm #1.0R (from_seq l s);
}

inline_for_extraction noextract
fn gpu_matrix_to_array
  (#et:Type0) {| sized et |}
  (#mrows #mcols #brows #bcols : SZ.t)
  (#l : mlayout4 mrows mcols brows bcols)
  (a : vec et)
  (gm : gpu_matrix et l)
  (#s : erased (seq et){Seq.length s == mlayout_size l})
  (#em : _)
  preserves
    (gm |-> em) **
    cpu
  requires
    (* same *)
    pure (mlayout_size l > 0) **
    (a |-> s)
  ensures
    pure (SZ.fits (mlayout_size l) /\ Pulse.Lib.Vec.length a == (mlayout_size l)) **
    (a |-> to_seq l em)
{
  Pulse.Lib.Vec.pts_to_len a;
  open FStar.SizeT;
  unfold gpu_matrix_pts_to gm #1.0R em;
  let sz = (mrows *^ brows) *^ (mcols *^ bcols);
  A.varray_to_array #_ #_ #sz a gm;
  to_seq_rel l em;
  fold gpu_matrix_pts_to gm #1.0R em;
}
