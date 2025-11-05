module Kuiper.Matrix4
#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.GhostMap
open Kuiper.EMatrix4
open Kuiper.Injection { mk_cinj }
module A = Kuiper.VArray
module T = FStar.Tactics.V2
open FStar.SizeT { (/^), (%^), (+^), (-^), ( *^ )  }

inline_for_extraction noextract
type cit
  (#mrows #mcols #brows #bcols : erased nat)
  (l : mlayout4 mrows mcols brows bcols)
  : Type
  = szlt mrows & szlt mcols &
    szlt brows & szlt bcols

inline_for_extraction noextract
let clayout4_imap
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (c : clayout4 l)
  : cit l -> szlt ((mrows * brows) * (mcols * bcols))
  = fun (bi, bj, i, j) ->
      c.parent.c_to
        (s_undivmod c.c_brows (bi, i))
        (s_undivmod c.c_bcols (bj, j))

(* This is only between abstract indices and concrete indices.
   Nothing here depends on the *actual* layout. *)
#push-options "--split_queries always --retry 5 --z3rlimit 20" // flaky
inline_for_extraction noextract
let clayout4_bij
  (#mrows #mcols #brows #bcols : nat)
  (l : mlayout4 mrows mcols brows bcols)
  (_ : squash (SZ.fits ((mrows * brows) * (mcols * bcols))))
  : erased (natlt (mrows * brows) & natlt (mcols * bcols) =~ cit l)
= {
    ff = (fun (i, j) ->
             let i : natlt (mrows * brows) = i in
             let j : natlt (mcols * bcols) = j in
             (SZ.uint_to_t <| i / brows,
              SZ.uint_to_t <| j / bcols,
              SZ.uint_to_t <| i % brows,
              SZ.uint_to_t <| j % bcols) <: cit l);
    gg = (fun (bi, bj, i, j) ->
            let bi : szlt mrows = bi in
            let bj : szlt mcols = bj in
            let i  : szlt brows = i in
            let j  : szlt bcols = j in
            (SZ.v bi * brows + SZ.v i,
             SZ.v bj * bcols + SZ.v j));
    ff_gg = (fun (bi,bj,i,j) -> ());
    gg_ff = (fun (i,j) -> ());
}
#pop-options

// FIXME: The VC for this definition is huge. It's incredible
// we can actually print it out and solve it. Try to make
// sense of it and report bug in F*.
#push-options "--z3rlimit 50 --split_queries always"
inline_for_extraction noextract
instance cview_from_clayout4
  (et : Type)
  (#mrows #mcols : erased nat)
  (#brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (c : clayout4 l)
  : IView.ciview (aview_from_mlayout et l).iview =
{
  clen = c.parent.m_rows *^ c.parent.m_cols;

  sch = {
    cit = cit l;
    bij = clayout4_bij l ();
  };

  step = {
    cimap = mk_cinj (clayout4_imap c) #(fun idx1 idx2 -> ());
    compat = ez;
  };
}
#pop-options

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
  = ()

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

#push-options "--z3rlimit 40"
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
  assert pure (Seq.equal (to_seq l em) (A.to_seq (aview_from_mlayout et l) em));
  a'
}
#pop-options

#restart-solver // work around z3 crash
#push-options "--z3seed 2 --z3rlimit 40"
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
  // FIXME: rewrite each can't prove equality?
  // rewrite each to_seq l em
  //           as A.to_seq (aview_from_mlayout et l) em;
  rewrite
    p |-> Frac f (to_seq l em)
  as
    p |-> Frac f (A.to_seq (aview_from_mlayout et l) em);
  A.varray_abs (aview_from_mlayout et l) p;
  fold gpu_matrix_pts_to (from_array l p) #f em;
}
#pop-options

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
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  (#em : _)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    forall+ (_:natlt k). gpu_matrix_pts_to gm #(f /. k) em
{
  unfold gpu_matrix_pts_to gm #f em;
  A.varray_share_n gm k;
  forevery_map
    (fun (i:natlt k) ->
      A.varray_pts_to gm #(f /. k) em)
    (fun (i:natlt k) ->
      gpu_matrix_pts_to gm #(f /. k) em)
    fn i {
      fold gpu_matrix_pts_to gm #(f /. k) em;
    };
}

ghost
fn gpu_matrix_gather_n
  (#et:Type0)
  (#mrows #mcols #brows #bcols : erased nat)
  (#l : mlayout4 mrows mcols brows bcols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  (#em : _)
  requires
    forall+ (_:natlt k). gpu_matrix_pts_to gm #(f /. k) em
  ensures
    gpu_matrix_pts_to gm #f em
{
  forevery_map
    (fun (i:natlt k) ->
      gpu_matrix_pts_to gm #(f /. k) em)
    (fun (i:natlt k) ->
      A.varray_pts_to gm #(f /. k) em)
    fn i{
      unfold gpu_matrix_pts_to gm #(f /. k) em;
    };
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


#push-options "--z3rlimit 80"
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
  assert (pure (brows > 0));
  assert (pure (bcols > 0));
  let v = A.varray_read_cell' gm (bi, bj, i, j) i_low;
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
      as A.ci_to_ai (aview_from_mlayout et l) (bi, bj, i, j);
  A.varray_write_cell gm (bi, bj, i, j) v1;
  // admit(); // This function is very flaky and sometimes its verification loops.
  // Seems to be working now... but admit if this fails again.
  with i1 lv1.
    assert (A.varray_pts_to_cell gm i1 lv1);
  rewrite A.varray_pts_to_cell gm i1 lv1
       as gpu_matrix_pts_to_cell gm bi bj i j v1;
  ();
}

#push-options "--z3rlimit 40 --split_queries always"
let bij_2_4
  (#mrows #mcols #brows #bcols : nat)
  : (natlt (mrows * brows) & natlt (mcols * bcols) =~ natlt mrows & natlt mcols & natlt brows & natlt bcols)
= mk_bijection
    #(natlt (mrows * brows) & natlt (mcols * bcols))
    #(natlt mrows & natlt mcols & natlt brows & natlt bcols)
    (fun (r,c) ->
       (r / brows,
        c / bcols,
        r % brows,
        c % bcols))
    (fun (br, bc, i, j) ->
       (br * brows + i,
        bc * bcols + j))
    ez
    ez
#pop-options

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
    pure (SZ.fits (mlayout_size l))
  ensures
    forall+ bi bj i j.
      gpu_matrix_pts_to_cell gm #f bi bj i j (macc em bi bj i j)
{
  unfold gpu_matrix_pts_to gm #f em;
  A.varray_pts_to_ref gm;
  A.varray_explode gm;
  forevery_rw_type
    (aview_from_mlayout et l).iview.sch.ait
    (natlt (mrows * brows) & natlt (mcols * bcols))
    (fun rc ->
      A.varray_pts_to_cell gm #f rc ((aview_from_mlayout et l).igm.acc em rc));
  forevery_iso bij_2_4 _;
  forevery_ext
    _
    (fun (bi,bj,i,j) ->
      gpu_matrix_pts_to_cell gm #f bi bj i j (macc em bi bj i j));
  forevery_unflatten4' _;
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
    pure (SZ.fits (mlayout_size l))
  requires
    forall+ bi bj i j.
      gpu_matrix_pts_to_cell gm #f bi bj i j (macc em bi bj i j)
  ensures
    gpu_matrix_pts_to gm #f em
{
  forevery_flatten4' (fun (bi,bj,i,j) ->
    gpu_matrix_pts_to_cell gm #f bi bj i j (macc em bi bj i j));
  forevery_iso (bij_sym bij_2_4) _;
  forevery_ext
    _
    (fun rc ->
      A.varray_pts_to_cell gm #f rc ((aview_from_mlayout et l).igm.acc em rc));
  A.varray_implode gm;
  fold gpu_matrix_pts_to gm #f em;
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
    a |-> s **
    cpu
  requires
    (* silly, but this shows that the multiplication below does not overflow.
    If we had a mul_underspec, we would not need this, I think. *)
    pure (mlayout_size l > 0) **
    gm |-> em
  ensures
    pure (SZ.fits (mlayout_size l) /\ Pulse.Lib.Vec.length a == (mlayout_size l)) **
    (gm |-> from_seq l s)
{
  Pulse.Lib.Vec.pts_to_len a;
  assert (pure (SZ.fits (mlayout_size l)));
  unfold gpu_matrix_pts_to gm #1.0R em;
  let sz = (mrows *^ brows) *^ (mcols *^ bcols);
  A.varray_from_array #_ #_ sz gm a;
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
    gm |-> em **
    cpu
  requires
    (* same *)
    pure (mlayout_size l > 0) **
    a |-> s
  ensures
    pure (SZ.fits (mlayout_size l) /\ Pulse.Lib.Vec.length a == (mlayout_size l)) **
    (a |-> to_seq l em)
{
  Pulse.Lib.Vec.pts_to_len a;
  open FStar.SizeT;
  unfold gpu_matrix_pts_to gm #1.0R em;
  let sz = (mrows *^ brows) *^ (mcols *^ bcols);
  A.varray_to_array #_ #_ sz a gm;
  to_seq_rel l em;
  fold gpu_matrix_pts_to gm #1.0R em;
}
