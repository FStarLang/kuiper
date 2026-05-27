module Kuiper.IArray
inline_for_extraction noextract let x = 1
#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.IView
module T = FStar.Tactics.V2
module SZ = Kuiper.SizeT

let oplus (#a #b : Type) (f : a -> GTot b) (x : a) (y : b) : a -> GTot b =
  fun x' ->
    if t2b (x == x')
    then y
    else f x'

inline_for_extraction
val iarray (et : Type0) (vw : aiview) : Type0

val is_global_iarray (#et : Type0) (#vw : aiview) (arr : iarray et vw) : prop

inline_for_extraction noextract
val from_array
  (#et : Type0)
  (vw : aiview)
  (arr : larray et (len vw))
  : iarray et vw

inline_for_extraction noextract
val core
  (#et : Type0)
  (#vw : aiview)
  (g : iarray et vw)
  : larray et (len vw)

val lem_from_array_core
  (#et : Type0)
  (#vw : aiview)
  (arr : iarray et vw)
  : Lemma (ensures from_array vw (core arr) == arr)
          [SMTPat (core arr)]

val lem_core_from_array
  (#et : Type0)
  (vw : aiview)
  (p : larray et (len vw))
  : Lemma (ensures core (from_array vw p) == p)
          [SMTPat (from_array vw p)]

(* Ownership over a single index. *)
val iarray_pts_to_cell
  (#et:Type)
  (#vw : aiview)
  ([@@@mkey] a : iarray et vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.ait)
  (v : et)
  : slprop

val iarray_pts_to_cell_def
  (#et : Type)
  (#vw : aiview)
  (a : iarray et vw)
  (#f : perm)
  (i : vw.ait)
  (v : et)
  : Lemma (iarray_pts_to_cell a #f i v ==
            pts_to_cell (core a) #f (it_to_nat vw i) v)

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#vw : aiview)
  : has_pts_to (cell (iarray et vw) vw.ait) et
= {
  pts_to = (fun (Cell ar i) #f v -> iarray_pts_to_cell #et #vw ar #f i v);
}

val iarray_pts_to
  (#et:Type0) (#vw : aiview)
  ([@@@mkey] a : iarray et vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : (vw.ait -> GTot et))
  : slprop

instance
val is_send_across_global_iarray
  (#et:Type0)
  (#vw : aiview)
  (x: iarray et vw { is_global_iarray x })
  (#f : perm)
  (v : (vw.ait -> GTot et))
  : is_send_across gpu_of (iarray_pts_to x #f v)

instance
val is_send_across_global_iarray_cell
  (#et:Type0)
  (#vw : aiview)
  (a: iarray et vw { is_global_iarray a })
  (#f : perm)
  (i : vw.ait)
  (v : et)
  : is_send_across gpu_of (iarray_pts_to_cell a #f i v)

unfold
instance has_pts_to (#et:Type0) (#vw : aiview)
  : has_pts_to (iarray et vw) (vw.ait -> GTot et) = {
  pts_to = iarray_pts_to;
}

ghost
fn iarray_pts_to_ref
  (#et:Type0)
  (#vw : aiview)
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.ait -> GTot et))
  preserves
    a |-> Frac f v
  ensures
    pure (SZ.fits (len vw))

(* The function on the RHS is extensional. *)
ghost
fn iarray_ext
  (#et:Type)
  (#vw : aiview)
  (a : iarray et vw)
  (#f : perm)
  (v1 v2 : (vw.ait -> GTot et))
  requires pure (forall x. v1 x == v2 x)
  requires a |-> Frac f v1
  ensures  a |-> Frac f v2

ghost
fn iarray_explode
  (#et:Type)
  (#vw : aiview)
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.ait -> GTot et))
  requires
    a |-> Frac f v
  ensures
    forall+ (i : vw.ait).
      Cell a i |-> Frac f (v i)

ghost
fn iarray_implode
  (#et:Type)
  (#vw : aiview)
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.ait -> GTot et))
  requires pure (SZ.fits (len vw))
  requires
    forall+ (i : vw.ait).
      Cell a i |-> Frac f (v i)
  ensures
    a |-> Frac f v

let g_seq_acc (#a:Type) (#len:nat)
  (s : lseq a len)
  (i : natlt len)
  : GTot a = s @! i

(* Begin viewing something abstractly, with the trivial view. *)
ghost
fn iarray_begin_
  (#et : Type0)
  (#len : erased nat)
  (a : larray et len)
  (#f : perm)
  (#v : lseq et len)
  requires
    a |-> Frac f v
  ensures
    from_array (raw_view #len) a |-> Frac f (g_seq_acc v)

inline_for_extraction noextract
fn iarray_begin
  (#et : Type0)
  (#len : erased nat)
  (a : larray et len)
  (#f : perm)
  (#v : erased (lseq et len))
  requires
    a |-> Frac f v
  returns
    va : iarray et (raw_view #len)
  ensures
    va |-> Frac f (g_seq_acc v)

ghost
fn iarray_end_
  (#et:Type0)
  (#len : erased nat)
  (a : iarray et (raw_view #len))
  (#f : perm)
  (#v : natlt len -> GTot et)
  requires
    a |-> Frac f v
  ensures
    core a |-> Frac f (Seq.init_ghost len v)

inline_for_extraction noextract
fn iarray_end
  (#et:Type0)
  (#len : erased nat)
  (a : iarray et (raw_view #len))
  (#f : perm)
  (#v : natlt len -> GTot et)
  requires
    a |-> Frac f v
  returns
    a' : larray et len
  ensures
    a' |-> Frac f (Seq.init_ghost len v)

inline_for_extraction noextract
fn iarray_end2
  (#et : Type0)
  (#vw : aiview { is_full_view vw })
  (a : iarray et vw)
  (#f : perm)
  (#v : vw.ait -> GTot et)
  requires
    a |-> Frac f v
  returns
    a' : larray et (len vw)
  ensures
    a' |-> Frac f (Seq.init_ghost (len vw) (fun (i : natlt (len vw)) -> v (it_of_nat vw i)))

ghost
fn iarray_cell_reindex
  (#et : Type0)
  (#f : perm)
  (#vw #vw' : aiview)
  (a : iarray et vw)
  (i : vw.ait)
  (a' : iarray et vw')
  (i' : vw'.ait)
  (#v: et)
  requires
    pure (vw.len == vw'.len /\ core a == core a')
  requires
    pure (it_to_nat vw i == it_to_nat vw' i')
  requires
    Cell a i |-> Frac f v
  ensures
    Cell a' i' |-> Frac f v

ghost
fn iarray_reindex_
  (#et : Type0)
  (#vw : aiview)
  (#ait' : Type)
  (bij : vw.ait =~ ait')
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.ait -> GTot et))
  requires
    a |-> Frac f v
  ensures
    from_array (reindex_view vw bij) (core a) |-> Frac f (v `oo` bij.gg)

inline_for_extraction noextract
unobservable
fn iarray_reindex
  (#et : Type0)
  (#vw : aiview)
  (#ait' : Type)
  (bij : vw.ait =~ ait')
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.ait -> GTot et))
  requires
    a |-> Frac f v
  returns
    va : iarray et (reindex_view vw bij)
  ensures
    va |-> Frac f (v `oo` bij.gg)

ghost
fn iarray_split2_
  (#et:Type0)
  (vw1 vw2 : aiview { len vw1 == len vw2 }) // needed?
  (#_ : squash (no_overlap vw1.step.imap.f vw2.step.imap.f))
  (a : iarray et (sum_aiview vw1 vw2 #())) /// argh!!!! affects typeclass resolution!!!!
  (#f : perm)
  (#v : either vw1.ait vw2.ait -> GTot et)
  requires
    a |-> Frac f v
  ensures
    (from_array vw1 (core a) |-> Frac f (v `oo` Inl)) **
    (from_array vw2 (core a) |-> Frac f (v `oo` Inr))

inline_for_extraction noextract
fn iarray_split2
  (#et:Type0)
  (vw1 vw2 : aiview { len vw1 == len vw2 }) // needed?
  (#_ : squash (no_overlap vw1.step.imap.f vw2.step.imap.f))
  (a : iarray et (sum_aiview vw1 vw2 #())) /// argh!!!! affects typeclass resolution!!!!
  (#f : perm)
  (#v : either vw1.ait vw2.ait -> GTot et)
  requires
    a |-> Frac f v
  returns
    a1a2 : iarray et vw1 & iarray et vw2
  ensures
    (fst a1a2 |-> Frac f (v `oo` Inl)) **
    (snd a1a2 |-> Frac f (v `oo` Inr))

ghost
fn iarray_split_n
  (#et : Type0)
  (#n : pos)
  (vw : natlt n -> aiview { forall i. len (vw i) = len (vw 0) })
  (#_ : squash (no_overlap_fam n vw))
  (a : iarray et (sum_aiview_fam n vw #()))
  (#f : perm)
  (#v : (i:natlt n & (vw i).ait) -> GTot et)
  requires
    a |-> Frac f v
  ensures
    forall+ (i : natlt n).
      from_array (vw i) (core a) |-> Frac f (fun j -> v (| i, j |))

ghost
fn iarray_share_n
  (#et:Type0)
  (#vw : aiview)
  (a : iarray et vw)
  (k : pos)
  (#f : perm)
  (#v : vw.ait -> GTot et)
  requires
    a |-> Frac f v
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) v

ghost
fn iarray_gather_n
  (#et:Type0)
  (#vw : aiview)
  (a : iarray et vw)
  (k : pos)
  (#f : perm)
  (#v : vw.ait -> GTot et)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) v
  ensures
    a |-> Frac f v

ghost
fn iarray_pts_to_eq
  (#et:Type0)
  (#vw : aiview)
  (a : iarray et vw)
  (#f1 f2 : perm)
  (#v1 #v2 : vw.ait -> GTot et)
  requires
    a |-> Frac f1 v1 **
    a |-> Frac f2 v2
  ensures
    a |-> Frac f1 v2 **
    a |-> Frac f2 v2

inline_for_extraction noextract
fn iarray_write_cell
  (#et:Type0)
  (#vw : aiview) {| cw : ciview vw |}
  (a : iarray et vw)
  (ci : cw.sch.cit)
  (v1 : et)
  (#v0 : erased et)
  requires
    Cell a (ci_to_ai vw ci) |-> v0
  ensures
    Cell a (ci_to_ai vw ci) |-> v1

inline_for_extraction noextract
fn iarray_write_cell'
  (#et:Type0)
  (#vw : aiview) {| cw : ciview vw |}
  (a : iarray et vw)
  (ai : erased vw.ait)
  (ci : cw.sch.cit)
  (v1 : et)
  (#v0 : erased et)
  requires
    (Cell a (reveal ai) |-> reveal v0) **
    pure (ai == ci_to_ai vw ci)
  ensures
    Cell a (reveal ai) |-> reveal v1

inline_for_extraction noextract
fn iarray_read_cell
  (#et:Type0)
  (#vw : aiview) {| cw : ciview vw |}
  (a : iarray et vw)
  (ci : cw.sch.cit)
  (#f : perm)
  (#v0 : erased et)
  requires
    iarray_pts_to_cell a #f (ci_to_ai vw ci) v0
  returns
    v : et
  ensures
    iarray_pts_to_cell a #f (ci_to_ai vw ci) v **
    pure (v == v0)

inline_for_extraction noextract
fn iarray_read_cell'
  (#et:Type0)
  (#vw : aiview) {| cw : ciview vw |}
  (a : iarray et vw)
  (i : cw.sch.cit)
  (ai : erased vw.ait)
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
fn iarray_read
  (#et:Type0)
  (#vw : aiview) {| cw : ciview vw |}
  (a : iarray et vw)
  (ci : cw.sch.cit)
  (#f : perm)
  (#v : (vw.ait -> GTot et))
  preserves
    a |-> Frac f v
  returns
    e : et
  ensures
    pure (e == v (ci_to_ai vw ci))

inline_for_extraction noextract
fn iarray_write
  (#et:Type0)
  (#vw : aiview) {| cw : ciview vw |}
  (a : iarray et vw)
  (ci : cw.sch.cit)
  (e : et)
  (#v0 : (vw.ait -> GTot et))
  requires
    a |-> v0
  ensures
    a |-> oplus v0 (ci_to_ai vw ci) e
