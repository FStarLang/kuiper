module Kuiper.ArrayView
#lang-pulse

open Kuiper
open Kuiper.Bijection
module Enum = Kuiper.Enumerable
module B = Kuiper.Array (* base *)
module T = FStar.Tactics.V2
module SZ = FStar.SizeT

let to_from (#a:Type) (#len:nat) (#vt:Type)
  (vw : aview a len vt)
  (s : lseq a len)
  : Lemma (ensures to_seq vw (from_seq vw s) == s)
          [SMTPat (to_seq vw (from_seq vw s))]
  = let _ = vw.igm.bij.gg (F.on_g vw.it <| fun i -> s @! it_to_nat vw i) in
    (* funny, mentioning the term above (= from_seq vw s) makes the proof work. *)
    assert (Seq.equal s (to_seq vw (from_seq vw s)))

let to_seq_upd (#a:Type) (#len:nat) (#vt:Type)
  (vw : aview a len vt)
  (v : vt)
  (i : vw.it)
  (x : a)
  : Lemma (ensures to_seq vw (vw.igm.upd v i x) == Seq.upd (to_seq vw v) (it_to_nat vw i) x)
          [SMTPat (to_seq vw (vw.igm.upd v i x))]
  = assert (to_seq vw (vw.igm.upd v i x) `Seq.equal` Seq.upd (to_seq vw v) (it_to_nat vw i) x)

let varray #a #len #vt vw =
  B.gpu_array a len

let core a = a
let core_match a1 a2 = ()

let varray_pts_to
  (#et:Type) (#len : nat) (#vt:_) (#vw : aview et len vt)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : vt)
  : slprop
  =
    B.gpu_pts_to_array a #f (to_seq vw v)

inline_for_extraction noextract
fn varray_concr
  (#t:Type0)
  (#len : erased nat)
  (#vt:Type0)
  (#vw : aview t len vt)
  (a : varray vw)
  (#v : erased vt)
  requires
    a |-> v
  ensures
    core a |-> to_seq vw v
{
  unfold varray_pts_to a v;
}

inline_for_extraction noextract
fn varray_abs
  (#t:Type0)
  (#len0 : erased nat) (#vt0:Type0) (#vw0 : aview t len0 vt0)
  (a : varray vw0)
  (#len : erased nat) (#vt:Type0) (vw : aview t len vt)
  (#v : erased vt)
  requires
    core a |-> to_seq vw v
  returns
    a' : varray vw
  ensures
    pure (len0 == len /\ core a == core a') **
    (a' |-> v)
{
  gpu_pts_to_ref (core a);
  let a' : varray vw = core a;
  rewrite each core a as a';
  fold varray_pts_to #t #len #vt #vw a' #1.0R v;
  a'
}

inline_for_extraction noextract
fn varray_alloc0
  (#et:Type) {| sized et |}
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
  fold varray_pts_to #et #len #vt #vw a #1.0R v;
  a
}

// inline_for_extraction noextract
// fn varray_alloc1
//   (#et:Type) {| sized et |}
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
  (#et:Type)
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
  B.gpu_array_free a;
}

ghost
fn varray_share_n
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (#vw : aview et len vt)
  (#[T.exact (`0)] uid: int)
  (a : varray vw)
  (k : pos)
  (#f : perm)
  (#v : vt)
  requires
    varray_pts_to a #f v
  ensures
    bigstar #uid 0 k (fun _ -> varray_pts_to a #(f /. k) v)
{
  unfold varray_pts_to a #f v;
  B.gpu_slice_share #uid a 0 len k;
}

ghost
fn varray_gather_n
  (#et:Type)
  (#len : erased nat) (#vt:Type0) (#vw : aview et len vt)
  (#uid: int)
  (a : varray vw)
  (k : pos)
  (#f : perm)
  (#v : vt)
  requires
    bigstar #uid 0 k (fun _ -> varray_pts_to a #(f /. k) v)
  ensures
    varray_pts_to a #f v
{
  B.gpu_slice_gather #uid a 0 len k;
  fold varray_pts_to a #f v;
}

inline_for_extraction noextract
fn varray_read
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (#f : perm)
  (#v : erased vt)
  requires
    gpu **
    varray_pts_to a #f v
  returns
    e : et
  ensures
    gpu **
    varray_pts_to a #f v **
    pure (e == vw.igm.acc v (cit_to_it vw i))
{
  let ni = cidx cw i;
  unfold varray_pts_to a #f v;
  let r = B.gpu_array_read #et #len #0 #len a #f ni;
  fold varray_pts_to a #f v;
  r
}

inline_for_extraction noextract
fn varray_write
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (e : et)
  (#v0 : erased vt)
  requires
    gpu **
    (a |-> v0)
  ensures
    gpu **
    (a |-> vw.igm.upd v0 (cit_to_it vw i) e)
{
  let ci = cidx cw i;
  unfold varray_pts_to a v0;
  B.gpu_array_write #et #len #0 #len a ci e;
  fold varray_pts_to a (vw.igm.upd v0 (cit_to_it vw i) e);
}

let varray_pts_to_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt)
  ([@@@mkey] a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.it)
  (v : et)
  : slprop
  = gpu_pts_to_slice a #f (i |~> vw.ibij) ((i |~> vw.ibij) + 1) seq![v]

inline_for_extraction noextract
fn varray_read_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (#f : perm)
  (#v0 : erased et)
  requires
    gpu **
    varray_pts_to_cell a #f (cit_to_it vw i) v0
  returns
    v : et
  ensures
    gpu **
    varray_pts_to_cell a #f (cit_to_it vw i) v **
    pure (v == v0)
{
  let ci = cidx cw i;
  unfold varray_pts_to_cell a #f (cit_to_it vw i) v0;
  let r = B.gpu_array_read #et #len #ci #(ci+1) a #f ci;
  fold varray_pts_to_cell a #f (cit_to_it vw i) v0;
  r
}

inline_for_extraction noextract
fn varray_write_cell
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt) {| cw : cview vw |}
  (a : varray vw)
  (i : cw.cit)
  (v1 : et)
  (#v0 : erased et)
  requires
    gpu **
    varray_pts_to_cell a (cit_to_it vw i) v0
  ensures
    gpu **
    varray_pts_to_cell a (cit_to_it vw i) v1
{
  let ci = cidx cw i;
  unfold varray_pts_to_cell a (cit_to_it vw i) v0;
  B.gpu_array_write #_ #_ #ci #(ci+1) a ci v1;
  with s'. assert (B.gpu_pts_to_slice a ci (ci+1) s');
  Kuiper.Seq.Common.lem_one_elem s' v1;
  fold varray_pts_to_cell a (cit_to_it vw i) v1;
  ()
}

ghost
fn varray_explode
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt)
  {| enum : Enumerable.enumerable vw.it |}
  (a : varray vw)
  (#f : perm)
  (#v : vt)
  requires
    varray_pts_to a #f v
  ensures
    forall+ (i : vw.it).
      varray_pts_to_cell #et #len #vt #vw a #f i (vw.igm.acc v i)
{
  (* jeez *)
  unfold varray_pts_to a #f v;
  B.gpu_array_slice_1 a;
  Enumerable.bijection_implies_equal_cardinal
    vw.it (natlt len) vw.ibij;
  assert (pure (enum._cardinal == len));
  rewrite
    bigstar 0 len
      (fun i -> gpu_pts_to_slice a #f i (i + 1) seq![to_seq vw v @! i])
  as
    bigstar 0 (Enum.cardinal vw.it #_)
      (fun i -> gpu_pts_to_slice a #f i (i + 1) seq![to_seq vw v @! i]);
  forevery_fromstar #vw.it #enum
    (fun (i:vw.it) ->
      gpu_pts_to_slice a #f (Enum.to_nat i) ((Enum.to_nat i) + 1) seq![to_seq vw v @! (Enum.to_nat i)]);
  forevery_permute #vw.it #enum (vw.ibij `bij_comp` bij_sym enum.bij)
    (fun (i:vw.it) ->
      gpu_pts_to_slice a #f (Enum.to_nat i)
                            ((Enum.to_nat i) + 1)
                            seq![to_seq vw v @! Enum.to_nat i]);
  forevery_ext #vw.it
    (fun i ->
      gpu_pts_to_slice a #f (Enum.to_nat (enum.bij.gg (vw.ibij.ff i)))
                            ((Enum.to_nat (enum.bij.gg (vw.ibij.ff i))) + 1)
                            seq![to_seq vw v @! (Enum.to_nat (enum.bij.gg (vw.ibij.ff i)))])
    (fun i -> gpu_pts_to_slice a #f (i |~> vw.ibij) ((i |~> vw.ibij)+1) seq![vw.igm.acc v i]);
}

ghost
fn varray_implode
  (#et:Type)
  (#len : erased nat) (#vt:Type0)
  (#vw : aview et len vt)
  {| enum : Enumerable.enumerable vw.it |}
  (a : varray vw)
  (#f : perm)
  (#v : vt)
  requires
    forall+ (i : vw.it).
      varray_pts_to_cell #et #len #vt #vw a #f i (vw.igm.acc v i)
  ensures
    varray_pts_to a #f v
{
  Enumerable.bijection_implies_equal_cardinal
    vw.it (natlt len) vw.ibij;
  forevery_ext #vw.it
    (fun i -> gpu_pts_to_slice a #f (i |~> vw.ibij) ((i |~> vw.ibij)+1) seq![vw.igm.acc v i])
    (fun i ->
      gpu_pts_to_slice a #f (Enum.to_nat (enum.bij.gg (vw.ibij.ff i)))
                            ((Enum.to_nat (enum.bij.gg (vw.ibij.ff i))) + 1)
                            seq![to_seq vw v @! (Enum.to_nat (enum.bij.gg (vw.ibij.ff i)))]);
  forevery_permute_back #vw.it #enum (vw.ibij `bij_comp` bij_sym enum.bij)
    (fun (i:vw.it) ->
      gpu_pts_to_slice a #f (Enum.to_nat i)
                            ((Enum.to_nat i) + 1)
                            seq![to_seq vw v @! Enum.to_nat i]);
  forevery_tostar #vw.it #enum
    (fun (i:vw.it) ->
      gpu_pts_to_slice a #f (Enum.to_nat i) ((Enum.to_nat i) + 1) seq![to_seq vw v @! (Enum.to_nat i)]);
  rewrite
    bigstar 0 (Enum.cardinal vw.it #_)
      (fun i -> gpu_pts_to_slice a #f i (i + 1) seq![to_seq vw v @! i])
  as
    bigstar 0 len
      (fun i -> gpu_pts_to_slice a #f i (i + 1) seq![to_seq vw v @! i]);
  B.gpu_array_unslice_1 a;
  fold varray_pts_to a #f v;
}


inline_for_extraction noextract
fn varray_from_array
  (#et:Type) {| sized et |}
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
  unfold varray_pts_to va v;
  B.gpu_pts_to_ref va;
  B.gpu_memcpy_host_to_device va a len;
  fold varray_pts_to va (from_seq vw s);
}

inline_for_extraction noextract
fn varray_to_array
  (#et:Type) {| sized et |}
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
  unfold varray_pts_to va v;
  B.gpu_memcpy_device_to_host a va len;
  fold varray_pts_to va v;
}
