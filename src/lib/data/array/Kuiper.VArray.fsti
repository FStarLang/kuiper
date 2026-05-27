module Kuiper.VArray
inline_for_extraction noextract let x = 1
#lang-pulse

(* Virtual arrays, re-indexable and view-shiftable. *)

include Kuiper.View

open Kuiper
open Kuiper.Bijection
open Kuiper.View
module T = FStar.Tactics.V2
module SZ = Kuiper.SizeT
module F = FStar.FunctionalExtensionality

let view_equiv (#et #st : Type)
  (vw1 vw2 : aview et st)
  : prop
= vw1.iview.len == vw2.iview.len /\
  vw1.iview.ait == vw2.iview.ait /\
  F.feq_g vw1.iview.step.imap.f vw2.iview.step.imap.f /\
  (forall (x: st) (i: vw1.iview.ait). vw1.ctn.acc x i == vw2.ctn.acc x i) /\
  (* probably need more about the mappings in ctn *)
  True

inline_for_extraction
val varray (#a : Type0) (#st : Type0) (vw : aview a st) : Type0

val is_global_varray (#a : Type0) (#st : Type0) (#vw : aview a st) (_ : varray vw) : prop

inline_for_extraction noextract
val from_array
  (#a : Type0)
  (#st : Type)
  (vw : aview a st)
  (arr : larray a (len vw))
  : varray vw

inline_for_extraction noextract
val core
  (#a : Type)
  (#st : Type) (#vw : aview a st)
  (g : varray vw)
  : arr : larray a (len vw) { from_array vw arr == g }

val lem_from_array_core
  (#a : Type)
  (#st : Type) (#vw : aview a st)
  (arr : varray vw)
  : Lemma (ensures from_array vw (core arr) == arr)
          [SMTPat (core arr)]

val lem_core_from_array
  (#a : Type)
  (#st : Type) (#vw : aview a st)
  (p : larray a (len vw))
  : Lemma (ensures core (from_array vw p) == p)
          [SMTPat (from_array vw p)]

(* Ownership over a single index. *)
val varray_pts_to_cell
  (#et:Type)
  (#st:Type0) (#vw : aview et st)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.iview.ait)
  (v : et)
  : slprop

val varray_pts_to_cell_eq
  (#et:Type)
  (#st:Type0) (#vw : aview et st)
  (a : varray vw)
  (i : vw.iview.ait)
  (f : perm)
  (v : et)
  : Lemma (varray_pts_to_cell a #f i v
           ==
           pts_to_cell (core a) #f (vw.iview.step.imap.f i) v)

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#st : Type) (#vw : aview et st)
  : has_pts_to (cell (varray vw) vw.iview.ait) et
= {
  pts_to = (fun (Cell ar i) #f v -> varray_pts_to_cell #et #st #vw ar #f i v);
}

val varray_pts_to
  (#a:Type) (#st:_) (#vw : aview a st)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : st)
  : slprop

instance
val is_send_across_global_varray
  (#et:Type0)
  (#st : Type0)
  (#vw : aview et st)
  (x: varray vw { is_global_varray x })
  (#f : perm)
  (v : st)
  : is_send_across gpu_of (varray_pts_to x #f v)

instance
val is_send_across_global_varray_cell
  (#et:Type0)
  (#st : Type0)
  (#vw : aview et st)
  (a : varray vw { is_global_varray a })
  (#f : perm)
  (i : vw.iview.ait)
  (v : et)
  : is_send_across gpu_of (varray_pts_to_cell a #f i v)

unfold
instance has_pts_to (#a:Type) (#st:Type) (#vw : aview a st)
  : has_pts_to (varray vw) st = {
  pts_to = varray_pts_to;
}

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
    a' : larray et len
  ensures
    a' |-> Frac f v

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
  ensures pure (is_global_varray a)

inline_for_extraction noextract
fn varray_free
  (#et : Type0) (#st : Type0)
  (#vw : aview et st { is_full_view vw })
  (a : varray vw)
  (#v : erased st)
  preserves
    cpu
  requires
    on gpu_loc (a |-> v)
  ensures emp

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
    from_array vw1 (core a) |-> Frac f (fst v) **
    from_array vw2 (core a) |-> Frac f (snd v)

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
    fst a1a2 |-> Frac f (fst v) **
    snd a1a2 |-> Frac f (snd v) **
    pure (core (fst a1a2) == core a) **
    pure (core (snd a1a2) == core a)

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
    a |-> vw.ctn.upd v0 (ci_to_ai vw ci) e //TODO: consider rebinding this

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
    on gpu_loc (va |-> from_seq vw s)

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
