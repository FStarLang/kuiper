module Kuiper.ArrayView
#lang-pulse

open Kuiper
open Kuiper.Bijection
module Enum = Kuiper.Enumerable
module B = Kuiper.Array (* base *)
module T = FStar.Tactics.V2
module SZ = FStar.SizeT
module Trade = Pulse.Lib.Trade

(* Avoid ghost effect when using projector. *)
inline_for_extraction noextract
let cidx
  (#a : Type0) (#len : erased nat) (#vt : Type0)
  (#vw : aview a len vt) (cw : cview vw)
  (cit : cw.cit)
  : c:sz{c == cw.cibij.ff cit}
  = //cw.cibij.ff cit
  match cw with {cibij} -> cibij.ff cit

noeq
type varray (#a:Type0) (#len : erased nat) (#vt : Type0) (vw : aview a len vt) =
  | VA of B.gpu_array a len

inline_for_extraction noextract
let from_array
  (#a : Type0) (#len : erased nat) (#vt : Type0)
  (vw : aview a len vt)
  (arr : gpu_array a len)
  : varray vw
  = VA arr

let core (VA a) = a

let lem_from_array_core
  (#a : Type0)
  (#len : erased nat)
  (#vt : Type0) (#vw : aview a len vt)
  (arr : varray vw)
  : Lemma (ensures from_array vw (core arr) == arr)
          [SMTPat (core arr)]
  = ()

let lem_core_from_array
  (#a : Type0)
  (#len : erased nat)
  (#vt : Type0) (#vw : aview a len vt)
  (p : gpu_array a len)
  : Lemma (ensures core (from_array vw p) == p)
          [SMTPat (from_array vw p)]
  = ()

let varray_pts_to_cell
  (#et:Type0)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.it)
  (v : et)
  : slprop
  = gpu_pts_to_cell (core a) #f (i |~> vw.ibij) seq![v]

let varray_pts_to
  (#et:Type0) (#len : erased nat) (#vt:_) (#vw : aview et len vt)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : vt)
  : slprop
  =
    forall+ (i : vw.it).
      varray_pts_to_cell a #f i (vw.igm.acc v i)

ghost
fn __varray_abs
  (#et:Type0)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt)
  (a : varray vw)
  (#f : perm)
  (#v : vt)
  requires
    core a |-> Frac f (to_seq vw v)
  ensures
    forall+ (i : vw.it).
      varray_pts_to_cell #et #len #vt #vw a #f i (vw.igm.acc v i)
{
  B.gpu_array_slice_1 (core a);
  Enumerable.bijection_implies_equal_cardinal
    vw.it (natlt len) vw.ibij;
  assert (pure (vw.it_enum._cardinal == len));
  rewrite
    bigstar 0 len
      (fun i -> gpu_pts_to_slice (core a) #f i (i + 1) seq![to_seq vw v @! i])
  as
    bigstar 0 (Enum.cardinal vw.it #_)
      (fun i -> gpu_pts_to_slice (core a) #f i (i + 1) seq![to_seq vw v @! i]);
  forevery_fromstar #vw.it #vw.it_enum
    (fun (i:vw.it) ->
      gpu_pts_to_slice (core a) #f (Enum.to_nat i) ((Enum.to_nat i) + 1) seq![to_seq vw v @! (Enum.to_nat i)]);
  forevery_permute #vw.it #vw.it_enum (vw.ibij `bij_comp` bij_sym vw.it_enum.bij)
    (fun (i:vw.it) ->
      gpu_pts_to_slice (core a) #f (Enum.to_nat i)
                            ((Enum.to_nat i) + 1)
                            seq![to_seq vw v @! Enum.to_nat i]);
  forevery_ext #vw.it
    (fun i ->
      gpu_pts_to_slice (core a) #f (Enum.to_nat (vw.it_enum.bij.gg (vw.ibij.ff i)))
                            ((Enum.to_nat (vw.it_enum.bij.gg (vw.ibij.ff i))) + 1)
                            seq![to_seq vw v @! (Enum.to_nat (vw.it_enum.bij.gg (vw.ibij.ff i)))])
    (fun i -> gpu_pts_to_slice (core a) #f (i |~> vw.ibij) ((i |~> vw.ibij)+1) seq![vw.igm.acc v i]);
}


ghost
fn __varray_concr
  (#et:Type0)
  (#len : nat) (#vt:Type0)
  (#vw : aview et len vt)
  (a : varray vw)
  (#f : perm)
  (#v : vt)
  requires
    forall+ (i : vw.it).
      varray_pts_to_cell #et #len #vt #vw a #f i (vw.igm.acc v i)
  ensures
    core a |-> Frac f (to_seq vw v)
{
  Enumerable.bijection_implies_equal_cardinal
    vw.it (natlt len) vw.ibij;
  forevery_ext #vw.it
    (fun i -> gpu_pts_to_slice (core a) #f (i |~> vw.ibij) ((i |~> vw.ibij)+1) seq![vw.igm.acc v i])
    (fun i ->
      gpu_pts_to_slice (core a) #f (Enum.to_nat (vw.it_enum.bij.gg (vw.ibij.ff i)))
                            ((Enum.to_nat (vw.it_enum.bij.gg (vw.ibij.ff i))) + 1)
                            seq![to_seq vw v @! (Enum.to_nat (vw.it_enum.bij.gg (vw.ibij.ff i)))]);
  forevery_permute_back #vw.it #vw.it_enum (vw.ibij `bij_comp` bij_sym vw.it_enum.bij)
    (fun (i:vw.it) ->
      gpu_pts_to_slice (core a) #f (Enum.to_nat i)
                            ((Enum.to_nat i) + 1)
                            seq![to_seq vw v @! Enum.to_nat i]);
  forevery_tostar #vw.it #vw.it_enum
    (fun (i:vw.it) ->
      gpu_pts_to_slice (core a) #f (Enum.to_nat i) ((Enum.to_nat i) + 1) seq![to_seq vw v @! (Enum.to_nat i)]);
  rewrite
    bigstar 0 (Enum.cardinal vw.it #_)
      (fun i -> gpu_pts_to_slice (core a) #f i (i + 1) seq![to_seq vw v @! i])
  as
    bigstar 0 len
      (fun i -> gpu_pts_to_slice (core a) #f i (i + 1) seq![to_seq vw v @! i]);
  B.gpu_array_unslice_1 (core a);
}

ghost
fn varray_pts_to_ref
  (#t:Type0)
  (#len : erased nat)
  (#vt:Type0)
  (#vw : aview t len vt)
  (a : varray vw)
  (#f : perm)
  (#v : erased vt)
  preserves
    a |-> Frac f v
  ensures
    pure (SZ.fits len)
{
  unfold varray_pts_to a #f v;
  __varray_concr a;
  B.gpu_pts_to_ref (core a);
  __varray_abs a;
  fold varray_pts_to a #f v;
}

(* This is now trivial. *)
ghost
fn varray_explode
  (#et:Type0)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt)
  (a : varray vw)
  (#f : perm)
  (#v : vt)
  requires
    a |-> Frac f v
  ensures
    forall+ (i : vw.it).
      varray_pts_to_cell #et #len #vt #vw a #f i (vw.igm.acc v i)
{
  unfold varray_pts_to a #f v;
}

(* This is now trivial. *)
ghost
fn varray_implode
  (#et:Type0)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt)
  (a : varray vw)
  (#f : perm)
  (#v : vt)
  requires
    forall+ (i : vw.it).
      varray_pts_to_cell #et #len #vt #vw a #f i (vw.igm.acc v i)
  ensures
    a |-> Frac f v
{
  fold varray_pts_to a #f v;
}

ghost
fn varray_concr
  (#t:Type0)
  (#len : erased nat)
  (#vt:Type0)
  (#vw : aview t len vt)
  (a : varray vw)
  (#f : perm)
  (#v : erased vt)
  requires
    a |-> Frac f v
  ensures
    core a |-> Frac f (to_seq vw v)
{
  unfold varray_pts_to a #f v;
  __varray_concr a;
}

ghost
fn varray_abs
  (#t:Type0)
  (#len : erased nat)
  (#vt:Type0) (vw : aview t len vt)
  (a : gpu_array t len)
  (#f : perm)
  (#v : vt)
  requires
    a |-> Frac f (to_seq vw v)
  ensures
    from_array vw a |-> Frac f v
{
  let va = VA #t #len #vt #vw a;
  rewrite each a as core va;
  __varray_abs va;
  fold varray_pts_to #t #len #vt #vw va #f v;
  rewrite each va as from_array vw a;
}

ghost
fn varray_abs'
  (#t:Type0)
  (#len : erased nat)
  (#vt:Type0) (vw : aview t len vt)
  (a : gpu_array t len)
  (#f : perm)
  (#v : lseq t len)
  requires
    a |-> Frac f v
  ensures
    from_array vw a |-> Frac f (from_seq vw v)
{
  rewrite each v as to_seq vw (from_seq vw v);
  varray_abs vw a;
}

inline_for_extraction noextract
fn varray_alloc0
  (#et:Type0) {| sized et |}
  (len : SZ.t) (#vt:Type0) (vw : aview et len vt)
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
  with s.
    assert (a |-> s);
  let v = hide (from_seq vw s);
  rewrite each s as to_seq vw v;
  let va = VA #et #len #vt #vw a;
  rewrite each a as core va;
  __varray_abs va;
  fold varray_pts_to va v;
  va
}

// inline_for_extraction noextract
// fn varray_alloc1
//   (#et:Type0) {| sized et |}
//   (#len : SZ.t) (#vt:Type0) (vw : aview et len vt)
//   (v0 : vt)
//   preserves
//     cpu
//   requires
//     pure (SZ.fits len)
//   returns
//     a : varray vw
//   ensures
//     a |-> v0
// {
//   let a = varray_alloc0 #et len vw;
//   (* fill? *)
//   admit();
// }

inline_for_extraction noextract
fn varray_free
  (#et:Type0)
  (#len : erased nat) (#vt:Type0) (#vw : aview et len vt)
  (a : varray vw)
  (#v : erased vt)
  preserves
    cpu
  requires
    a |-> v
  ensures emp
{
  unfold varray_pts_to a v;
  __varray_concr a;
  B.gpu_array_free (core a);
}

ghost
fn varray_share_n
  (#et:Type0)
  (#len : erased nat) (#vt:Type0) (#vw : aview et len vt)
  (#[T.exact (`0)] uid: int)
  (a : varray vw)
  (k : pos)
  (#f : perm)
  (#v : vt)
  requires
    a |-> Frac f v
  ensures
    bigstar #uid 0 k (fun _ -> a |-> Frac (f /. k) v)
{
  (* Boring: share everything N-wise under the forall+, then commute
  the bigstar with forall+ *)
  admit();
}

ghost
fn varray_gather_n
  (#et:Type0)
  (#len : erased nat) (#vt:Type0) (#vw : aview et len vt)
  (#uid: int)
  (a : varray vw)
  (k : pos)
  (#f : perm)
  (#v : vt)
  requires
    bigstar #uid 0 k (fun _ -> a |-> Frac (f /. k) v)
  ensures
    a |-> Frac f v
{
  (* Boring: the reverse of above. *)
  admit();
}

// #set-options "--print_implicits"

inline_for_extraction noextract
fn varray_read
  (#et:Type0)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (#f : perm)
  (#v : erased vt)
  preserves
    gpu **
    (a |-> Frac f v)
  returns
    e : et
  ensures
    pure (e == vw.igm.acc v (cit_to_it vw i))
{
  let ni = cidx cw i;
  unfold varray_pts_to a #f v;
  forevery_extract (cit_to_it vw i) _;
  unfold varray_pts_to_cell a #f (cit_to_it vw i) (vw.igm.acc v (cit_to_it vw i));
  let r = B.gpu_array_read #et #len #_ #_ (core a) #f ni;
  fold varray_pts_to_cell a #f (cit_to_it vw i) (vw.igm.acc v (cit_to_it vw i));
  Trade.elim_trade _ _;
  fold varray_pts_to a #f v;
  r
}

inline_for_extraction noextract
fn varray_write
  (#et:Type0)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (e : et)
  (#v0 : erased vt)
  preserves
    gpu
  requires
    (a |-> v0)
  ensures
    (a |-> vw.igm.upd v0 (cit_to_it vw i) e)
{
  let ci = cidx cw i;
  unfold varray_pts_to a v0;
  forevery_extract_if (cit_to_it vw i)  _;
  unfold varray_pts_to_cell a (cit_to_it vw i) (vw.igm.acc v0 (cit_to_it vw i));
  B.gpu_array_write (core a) ci e;
  (* Should finish: there is conceptually trivial, but in practice
  very non-trivial rewrite to perform under the forall+. *)
  admit();
  // B.gpu_array_write #et #len #0 #len (core a) ci e;
  fold varray_pts_to a (vw.igm.upd v0 (cit_to_it vw i) e);
}

inline_for_extraction noextract
fn varray_read_cell
  (#et:Type0)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (#f : perm)
  (#v0 : erased et)
  preserves
    gpu
  requires
    varray_pts_to_cell a #f (cit_to_it vw i) v0
  returns
    v : et
  ensures
    varray_pts_to_cell a #f (cit_to_it vw i) v **
    pure (v == v0)
{
  let ci = cidx cw i;
  unfold varray_pts_to_cell a #f (cit_to_it vw i) v0;
  rewrite each (cit_to_it vw i |~> vw.ibij) as ci;
  let r = B.gpu_array_read #et #len #ci #(ci+1) (core a) #f (cidx cw i);
  rewrite each SZ.v ci as (cit_to_it vw i |~> vw.ibij);
  fold varray_pts_to_cell a #f (cit_to_it vw i) v0;
  r
}

inline_for_extraction noextract
fn varray_read_cell'
  (#et:Type0)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (ai : erased vw.it)
  (#f : perm)
  (#v0 : erased et)
  preserves
    gpu
  requires
    varray_pts_to_cell a #f ai v0 **
    pure (ai == cit_to_it vw i)
  returns
    v : et
  ensures
    varray_pts_to_cell a #f ai v **
    pure (v == v0)
{
  rewrite each ai as cit_to_it vw i;
  let res = varray_read_cell #et #len #vt a i;
  rewrite each cit_to_it vw i as ai;
  res
}

inline_for_extraction noextract
fn varray_write_cell
  (#et:Type0)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (v1 : et)
  (#v0 : erased et)
  preserves gpu
  requires
    varray_pts_to_cell a (cit_to_it vw i) v0
  ensures
    varray_pts_to_cell a (cit_to_it vw i) v1
{
  let ci = cidx cw i;
  unfold varray_pts_to_cell a (cit_to_it vw i) v0;
  rewrite each (cit_to_it vw i |~> vw.ibij) as ci;
  B.gpu_array_write #_ #_ #ci #(ci+1) (core a) ci v1;
  with s'. assert (B.gpu_pts_to_slice (core a) ci (ci+1) s');
  Kuiper.Seq.Common.lem_one_elem s' v1;
  rewrite each SZ.v ci as (cit_to_it vw i |~> vw.ibij);
  fold varray_pts_to_cell a (cit_to_it vw i) v1;
  ()
}

inline_for_extraction noextract
fn varray_write_cell'
  (#et:Type0)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (ai : erased vw.it)
  (v1 : et)
  (#v0 : erased et)
  preserves
    gpu
  requires
    varray_pts_to_cell a ai v0 **
    pure (ai == cit_to_it vw i)
  ensures
    varray_pts_to_cell a ai v1
{
  rewrite each ai as cit_to_it vw i;
  let res = varray_write_cell #et #len #vt a i v1;
  rewrite each cit_to_it vw i as ai;
  res
}


inline_for_extraction noextract
fn varray_from_array
  (#et:Type0) {| sized et |}
  (#len : SZ.t) (#vt:Type0)
  (#vw : aview et len vt)
  (va : varray vw)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == len})
  (#v : erased vt)
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

#set-options "--print_implicits"

inline_for_extraction noextract
fn varray_to_array
  (#et:Type0) {| sized et |}
  (#len : SZ.t) (#vt:Type0)
  (#vw : aview et len vt)
  (a : vec et)
  (va : varray vw)
  (#s : erased (seq et){Seq.length s == len})
  (#v : erased vt)
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
  from_to #et #len #vt vw v;
  rewrite each from_array vw (core va) as va;
  // rewrite each from_seq vw (to_seq vw v) as v;
  // ^ this fails in mysterious ways?!
  rewrite va |-> from_seq vw (to_seq vw v) as va |-> v;
  ();
}
