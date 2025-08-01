module Kuiper.IArray
inline_for_extraction noextract let x = 1
#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.IView
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

let oplus (#a #b : Type) (f : a -> GTot b) (x : a) (y : b) : a -> GTot b =
  fun x' ->
    if FStar.StrongExcludedMiddle.strong_excluded_middle (x == x')
    then y
    else f x'

new
inline_for_extraction
val iarray (et : Type0) (#len : erased nat) (vw : aiview len) : Type0

inline_for_extraction noextract
val from_array
  (#et : Type0)
  (#len : erased nat)
  (vw : aiview len)
  (arr : gpu_array et len)
  : iarray et vw

inline_for_extraction noextract
val core
  (#et : Type0)
  (#len : erased nat)
  (#vw : aiview len)
  (g : iarray et vw)
  : arr : Kuiper.Array.gpu_array et len { from_array vw arr == g }

val lem_from_array_core
  (#et : Type0)
  (#len : erased nat)
  (#vw : aiview len)
  (arr : iarray et vw)
  : Lemma (ensures from_array vw (core arr) == arr)
          [SMTPat (core arr)]

val lem_core_from_array
  (#et : Type0)
  (#len : erased nat)
  (vw : aiview len)
  (p : gpu_array et len)
  : Lemma (ensures core (from_array vw p) == p)
          [SMTPat (from_array vw p)]

(* Ownership over a single index. *)
val iarray_pts_to_cell
  (#et:Type)
  (#len : erased nat) (#vw : aiview len)
  ([@@@mkey] a : iarray et vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.ait)
  (v : et)
  : slprop

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#len : nat) (#vw : aiview len)
  : has_pts_to (cell (iarray et vw) vw.ait) et
= {
  pts_to = (fun (Cell ar i) #f v -> iarray_pts_to_cell #et #len #vw ar #f i v);
}

val iarray_pts_to
  (#et:Type0) (#len : erased nat) (#vw : aiview len)
  ([@@@mkey] a : iarray et vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : (vw.ait -> GTot et))
  : slprop

unfold
instance has_pts_to (#et:Type0) (#len : nat) (#vw : aiview len)
  : has_pts_to (iarray et vw) (vw.ait -> GTot et) = {
  pts_to = iarray_pts_to;
}

ghost
fn iarray_pts_to_ref
  (#et:Type0)
  (#len : erased nat)
  (#vw : aiview len)
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.ait -> GTot et))
  preserves
    a |-> Frac f v
  ensures
    pure (SZ.fits len)


(* The function on the RHS is extensional. *)
ghost
fn iarray_ext
  (#et:Type)
  (#len : erased nat)
  (#vw : aiview len)
  (a : iarray et vw)
  (#f : perm)
  (v1 v2 : (vw.ait -> GTot et))
  requires pure (forall x. v1 x == v2 x)
  requires a |-> Frac f v1
  ensures  a |-> Frac f v2

(* Note: the functions below use the Enumerable instance for vw.ait
   that is inside the aview record.
   We do this since it's
   not necessary for that enumeration to match the one in the typeclass system.
   For example, for a matrix view, that enumeration can be anything
   depending on the layout chosen, but the enumeration we want for the
   **abstract indices** is just lexicographic. *)

ghost
fn iarray_explode
  (#et:Type)
  (#len : erased nat)
  (#vw : aiview len)
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
  (#len : erased nat)
  (#vw : aiview len)
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.ait -> GTot et))
  requires pure (SZ.fits len)
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
  (#et:Type0)
  (#len : erased nat)
  (a : gpu_array et len)
  (#f : perm)
  (#v : lseq et len)
  requires
    a |-> Frac f v
  ensures
    from_array (raw_view #len) a |-> Frac f (g_seq_acc v)

inline_for_extraction noextract
fn iarray_begin
  (#et:Type0)
  (#len : erased nat)
  (a : gpu_array et len)
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
    a' : gpu_array et len
  ensures
    a' |-> Frac f (Seq.init_ghost len v)

inline_for_extraction noextract
fn iarray_end2
  (#et : Type0) (#len : erased nat)
  (#vw : aiview len { is_full_view vw })
  (a : iarray et vw)
  (#f : perm)
  (#v : vw.ait -> GTot et)
  requires
    a |-> Frac f v
  returns
    a' : gpu_array et len
  ensures
    a' |-> Frac f (Seq.init_ghost len (fun i -> v (it_of_nat vw i)))

ghost
fn iarray_reindex_
  (#et : Type0) (#len : nat)
  (#vw : aiview len)
  (#ait' : Type)
  {| Enumerable.enumerable ait' |}
  (bij : vw.ait =~ ait')
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.ait -> GTot et))
  requires
    a |-> Frac f v
  ensures
    from_array (reindex_view vw bij) (core a) |-> Frac f (v `oo` bij.gg)

inline_for_extraction noextract
fn iarray_reindex
  (#et : Type0) (#len : nat)
  (#vw : aiview len)
  (#ait' : Type)
  {| Enumerable.enumerable ait' |}
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
  (#et:Type0) (#len : nat)
  (vw1 vw2 : aiview len)
  (#_ : squash (no_overlap vw1.imap.f vw2.imap.f))
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
  (#et:Type0) (#len : nat)
  (vw1 vw2 : aiview len)
  (#_ : squash (no_overlap vw1.imap.f vw2.imap.f))
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
fn iarray_share_n
  (#et:Type0)
  (#len : erased nat) (#vw : aiview len)
  (#[T.exact (`0)] uid: int)
  (a : iarray et vw)
  (k : pos)
  (#f : perm)
  (#v : vw.ait -> GTot et)
  requires
    a |-> Frac f v
  ensures
    bigstar #uid 0 k (fun _ -> a |-> Frac (f /. k) v)

ghost
fn iarray_gather_n
  (#et:Type0)
  (#len : erased nat) (#vw : aiview len)
  (#[T.exact (`0)] uid: int)
  (a : iarray et vw)
  (k : pos)
  (#f : perm)
  (#v : vw.ait -> GTot et)
  requires
    bigstar #uid 0 k (fun _ -> a |-> Frac (f /. k) v)
  ensures
    a |-> Frac f v

inline_for_extraction noextract
fn iarray_write_cell
  (#et:Type0)
  (#len : erased nat)
  (#vw : aiview len) {| cw : cview vw |}
  (a : iarray et vw)
  (ci : cw.cit)
  (v1 : et)
  (#v0 : erased et)
  preserves gpu
  requires
    Cell a (ci_to_ai vw ci) |-> v0
  ensures
    Cell a (ci_to_ai vw ci) |-> v1

inline_for_extraction noextract
fn iarray_write_cell'
  (#et:Type0)
  (#len : erased nat)
  (#vw : aiview len) {| cw : cview vw |}
  (a : iarray et vw)
  (ai : erased vw.ait)
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

inline_for_extraction noextract
fn iarray_read_cell
  (#et:Type0)
  (#len : erased nat)
  (#vw : aiview len) {| cw : cview vw |}
  (a : iarray et vw)
  (ci : cw.cit)
  (#f : perm)
  (#v0 : erased et)
  preserves
    gpu
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
  (#len : erased nat)
  (#vw : aiview len) {| cw : cview vw |}
  (a : iarray et vw)
  (i : cw.cit)
  (ai : erased vw.ait)
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
fn iarray_read
  (#et:Type0)
  (#len : erased nat)
  (#vw : aiview len) {| cw : cview vw |}
  (a : iarray et vw)
  (ci : cw.cit)
  (#f : perm)
  (#v : (vw.ait -> GTot et))
  preserves
    gpu **
    (a |-> Frac f v)
  returns
    e : et
  ensures
    pure (e == v (ci_to_ai vw ci))

inline_for_extraction noextract
fn iarray_write
  (#et:Type0)
  (#len : erased nat)
  (#vw : aiview len) {| cw : cview vw |}
  (a : iarray et vw)
  (ci : cw.cit)
  (e : et)
  (#v0 : (vw.ait -> GTot et))
  preserves
    gpu
  requires
    a |-> v0
  ensures
    a |-> oplus v0 (ci_to_ai vw ci) e
