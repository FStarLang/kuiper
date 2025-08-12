module Kuiper.VArray
inline_for_extraction noextract let x = 1
#lang-pulse

(* Virtual arrays, re-indexable and view-shiftable. *)

include Kuiper.View

open Kuiper
open Kuiper.Bijection
open Kuiper.GhostMap { is_ghost_map }
open Kuiper.View
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

new
inline_for_extraction
val varray (#a : Type0) (#len : erased nat) (#st : Type0) (vw : aview a len st) : Type0

inline_for_extraction noextract
val from_array
  (#a : Type0)
  (#len : erased nat)
  (#st : Type)
  (vw : aview a len st)
  (arr : gpu_array a len)
  : varray vw

inline_for_extraction noextract
val core
  (#a : Type)
  (#len : erased nat)
  (#st : Type) (#vw : aview a len st)
  (g : varray vw)
  : arr : Kuiper.Array.gpu_array a len { from_array vw arr == g }

val lem_from_array_core
  (#a : Type)
  (#len : erased nat)
  (#st : Type) (#vw : aview a len st)
  (arr : varray vw)
  : Lemma (ensures from_array vw (core arr) == arr)
          [SMTPat (core arr)]

val lem_core_from_array
  (#a : Type)
  (#len : erased nat)
  (#st : Type) (#vw : aview a len st)
  (p : gpu_array a len)
  : Lemma (ensures core (from_array vw p) == p)
          [SMTPat (from_array vw p)]

(* Ownership over a single index. *)
val varray_pts_to_cell
  (#et:Type)
  (#len : erased nat) (#st:Type0) (#vw : aview et len st)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.iview.sch.ait)
  (v : et)
  : slprop

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#len : nat) (#st : Type) (#vw : aview et len st)
  : has_pts_to (cell (varray vw) vw.iview.sch.ait) et
= {
  pts_to = (fun (Cell ar i) #f v -> varray_pts_to_cell #et #len #st #vw ar #f i v);
}

val varray_pts_to
  (#a:Type) (#len : erased nat) (#st:_) (#vw : aview a len st)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : st)
  : slprop

unfold
instance has_pts_to (#a:Type) (#len : nat) (#st:Type) (#vw : aview a len st)
  : has_pts_to (varray vw) st = {
  pts_to = varray_pts_to;
}

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


(* Note: the functions below use the Enumerable instance for vw.iview.sch.ait
   that is inside the aview record.
   We do this since it's
   not necessary for that enumeration to match the one in the typeclass system.
   For example, for a matrix view, that enumeration can be anything
   depending on the layout chosen, but the enumeration we want for the
   **abstract indices** is just lexicographic. *)

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
    forall+ (i : vw.iview.sch.ait).
      Cell a i |-> Frac f (vw.igm.acc v i)

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
    forall+ (i : vw.iview.sch.ait).
      Cell a i |-> Frac f (vw.igm.acc v i)
  ensures
    a |-> Frac f v

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

(* Note how the spec type does not change at all. The mapping
is hidden in the view. *)
ghost
fn varray_reindex_
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  (#ait' : Type)
  {| Enumerable.enumerable ait' |}
  (bij : vw.iview.sch.ait =~ ait')
  (a : varray vw)
  (#f : perm)
  (#v : st)
  requires
    a |-> Frac f v
  ensures
    from_array (reindex_view vw bij) (core a) |-> Frac f v

inline_for_extraction noextract
fn varray_reindex
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  (#ait' : Type)
  {| Enumerable.enumerable ait' |}
  (bij : vw.iview.sch.ait =~ ait')
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

ghost
fn varray_split2_
  (#et : Type0) (#len : nat) (#st1 #st2 : Type)
  (vw1 : aview et len st1)
  (vw2 : aview et len st2)
  (#_ : squash (no_overlap vw1.iview.step.imap.f vw2.iview.step.imap.f))
  (a : varray (sum_aview vw1 vw2 #())) /// argh!!!! affects typeclass resolution!!!!
  (#f : perm)
  (#v : st1 & st2)
  requires
    a |-> Frac f v
  ensures
    (from_array vw1 (core a) |-> Frac f (fst v)) **
    (from_array vw2 (core a) |-> Frac f (snd v))

inline_for_extraction noextract
fn varray_split2
  (#et : Type0) (#len : nat) (#st1 #st2 : Type)
  (vw1 : aview et len st1)
  (vw2 : aview et len st2)
  (#_ : squash (no_overlap vw1.iview.step.imap.f vw2.iview.step.imap.f))
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

ghost
fn varray_join2_
  (#et : Type0) (#len : nat) (#st1 #st2 : Type)
  (#vw1 : aview et len st1)
  (#vw2 : aview et len st2)
  (#_ : squash (no_overlap vw1.iview.step.imap.f vw2.iview.step.imap.f))
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

inline_for_extraction noextract
fn varray_join2
  (#et : Type0) (#len : nat) (#st1 #st2 : Type)
  (#vw1 : aview et len st1)
  (#vw2 : aview et len st2)
  (#_ : squash (no_overlap vw1.iview.step.imap.f vw2.iview.step.imap.f))
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

inline_for_extraction noextract
fn varray_write_cell
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  {| cw : IView.ciview vw.iview |} // fixme
  (a : varray vw)
  (ci : cw.sch.cit)
  (v1 : et)
  (#v0 : erased et)
  preserves gpu
  requires
    Cell a (ci_to_ai vw ci) |-> v0
  ensures
    Cell a (ci_to_ai vw ci) |-> v1

inline_for_extraction noextract
fn varray_write_cell'
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  {| cw : IView.ciview vw.iview |} // fixme
  (a : varray vw)
  (ai : erased vw.iview.sch.ait)
  (ci : cw.sch.cit)
  (v1 : et)
  (#v0 : erased et)
  preserves
    gpu
  requires
    (Cell a (reveal ai) |-> reveal v0) **
    pure (ai == ci_to_ai vw ci)
  ensures
    Cell a (reveal ai) |-> reveal v1

inline_for_extraction noextract
fn varray_read_cell
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  {| cw : IView.ciview vw.iview |} // fixme
  (a : varray vw)
  (ci : cw.sch.cit)
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

inline_for_extraction noextract
fn varray_read_cell'
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  {| cw : IView.ciview vw.iview |} // fixme
  (a : varray vw)
  (i : cw.sch.cit)
  (ai : erased vw.iview.sch.ait)
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

inline_for_extraction noextract
fn varray_read
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  {| cw : IView.ciview vw.iview |} // fixme
  (a : varray vw)
  (ci : cw.sch.cit)
  (#f : perm)
  (#v : erased st)
  preserves
    gpu **
    (a |-> Frac f v)
  returns
    e : et
  ensures
    pure (e == vw.igm.acc v (ci_to_ai vw ci))

inline_for_extraction noextract
fn varray_write
  (#et : Type) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  {| cw : IView.ciview vw.iview |} // fixme
  (a : varray vw)
  (ci : cw.sch.cit)
  (e : et)
  (#v0 : erased st)
  preserves
    gpu
  requires
    a |-> v0
  ensures
    a |-> vw.igm.upd v0 (ci_to_ai vw ci) e

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
