module Kuiper.Array3
friend Kuiper.Array2
#lang-pulse

open Kuiper
open Kuiper.Chest
open Kuiper.Bijection
open Kuiper.Index
open Kuiper.Seq.Common { (@!) }
module T = Kuiper.Tensor
module Array2 = Kuiper.Array2
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

inline_for_extraction noextract
let adapt_cit_back (d0 d1 d2 : erased nat) (idx : raw_cit{cit_fits d0 d1 d2 idx}) : conc (desc d0 d1 d2) =
  match idx with
  | (i, j, k) -> (i, (j, (k, ())))

#push-options "--ifuel 3" // sigh
let abs_bij (#d0 #d1 #d2 : nat) : (abs (desc d0 d1 d2) =~ (ait d0 d1 d2)) =
  {
    ff = (fun (i, (j, (k, ()))) -> (i, j, k));
    gg = (fun (i, j, k) -> (i, (j, (k, ()))));
    ff_gg = ez;
    gg_ff = ez;
  }
#pop-options

let t (et : Type0) (#d0 #d1 #d2 : nat) (l : layout d0 d1 d2) : Type0 =
  T.tensor et l

let is_global (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  : prop =
  T.is_global a

let from_array
  (#et : Type0) (#d0 #d1 #d2 : erased nat)
  (l : layout d0 d1 d2)
  (a : larray et (layout_size l))
  : t et l
  = T.from_array _ a

let core
  (#et : Type0) (#d0 #d1 #d2 : erased nat) (#l : layout d0 d1 d2)
  (a : t et l)
  : larray et (layout_size l)
  = T.core a

let lem_core_from_array
  (#et : Type) (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]
   = ()

let lem_from_array_core
  (#et : Type) (#d0 #d1 #d2 : erased nat)
  (l : layout d0 d1 d2)
  (p : larray et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let lem_is_global_iff_core
  (#et : Type0) (#d0 #d1 #d2 : nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]
  = ()

let pts_to
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : EMatrix3.t et d0 d1 d2)
  : slprop
  = T.tensor_pts_to a #f s

instance is_send_across_global
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l { is_global a })
  (#f : perm) (s : EMatrix3.t et d0 d1 d2)
  : is_send_across gpu_of (pts_to a #f s)
  = solve

ghost
fn pts_to_ref
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm) (#s : erased (EMatrix3.t et d0 d1 d2))
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (layout_size l))
{
  unfold pts_to a #f s;
  T.tensor_pts_to_ref a;
  fold pts_to a #f s;
}

ghost
fn pts_to_ref_located
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#loc : _)
  (#f : perm) (#s : erased (EMatrix3.t et d0 d1 d2))
  preserves
    on loc (a |-> Frac f s)
  ensures
    pure (SZ.fits (layout_size l))
{
  ghost_impersonate loc
    (on loc (a |-> Frac f s))
    (on loc (a |-> Frac f s) ** pure (SZ.fits (layout_size l)))
    fn () {
      on_elim _;
      pts_to_ref a;
      on_intro (a |-> Frac f s);
    }
}

#push-options "--ifuel 3"
inline_for_extraction noextract
fn alloc0
  (#et:Type) {| sized et |}
  (d0 d1 d2 : szp)
  (l : layout d0 d1 d2 { is_full l })
  preserves
    cpu
  requires
    pure (SZ.fits (layout_size l))
  returns
    p : t et l
  ensures
    exists* em. on gpu_loc (p |-> em) **
    pure (is_full_array (core p))
  ensures
    pure (is_global p)
{
  let t = T.alloc0 #et (d0 *^ (d1 *^ d2)) l;
  with em. assert on gpu_loc (T.tensor_pts_to t em);
  rewrite on gpu_loc (T.tensor_pts_to t em)
       as on gpu_loc (pts_to t em);
  t
}
#pop-options

inline_for_extraction noextract
fn free
  (#et:Type)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2 { is_full l })
  (p : t et l)
  (#em : EMatrix3.t et d0 d1 d2)
  preserves
    cpu
  requires
    on gpu_loc (p |-> em) **
    pure (is_full_array (core p))
  ensures emp
{
  rewrite on gpu_loc (pts_to p em)
       as on gpu_loc (T.tensor_pts_to p em);
  T.free p;
}

ghost
fn lower
  (#et:Type)
  (#d0 #d1 #d2 : nat)
  (#l : layout d0 d1 d2 { is_full l })
  (g : t et l)
  (#s : EMatrix3.t et d0 d1 d2)
  (#f : perm)
  requires
    g |-> Frac f s
  ensures
    core g |-> Frac f (to_seq l s)
{
  unfold pts_to g #f s;
  T.tensor_concr g;
}

ghost
fn raise
  (#et:Type)
  (#d0 #d1 #d2 : nat)
  (l : layout d0 d1 d2 { is_full l })
  (p : larray et (layout_size l))
  (#f : perm)
  (#s : EMatrix3.t et d0 d1 d2)
  requires
    p |-> Frac f (to_seq l s)
  ensures
    from_array l p |-> Frac f s
{
  T.tensor_abs l p;
  fold pts_to (from_array l p) #f s;
}

ghost
fn raise'
  (#et:Type)
  (#d0 #d1 #d2 : nat)
  (l : layout d0 d1 d2 { is_full l })
  (p : larray et (layout_size l))
  (#f : perm)
  (#s : lseq et (layout_size l))
  requires
    p |-> Frac f s
  ensures
    from_array l p |-> Frac f (from_seq l s)
{
  rewrite each s as to_seq l (from_seq l s);
  raise l p;
}

inline_for_extraction noextract
fn copy_from_vec
  (#et:Type0) {| sized et |}
  (#d0 #d1 #d2 : szp)
  (#l : layout d0 d1 d2 { is_full l })
  (gm : t et l)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == d0 * d1 * d2})
  (#em : EMatrix3.t et d0 d1 d2)
  preserves
    cpu ** a |-> s
  requires
    on gpu_loc (gm |-> em)
  ensures
    on gpu_loc (gm |-> from_seq l s)
{
  map_loc gpu_loc
    #(gm |-> em)
    #(core gm |-> to_seq l em)
    fn () { lower gm; };
  Pulse.Lib.Vec.pts_to_len a;
  gpu_memcpy_host_to_device (core gm) a (d0 *^ (d1 *^ d2));
  map_loc gpu_loc
    #(core gm |-> s)
    #(gm |-> from_seq l s)
    fn () {
      raise' l (core gm);
      rewrite from_array l (core gm) |-> Frac 1.0R (from_seq l s)
           as gm |-> from_seq l s;
    };
}

inline_for_extraction noextract
fn copy_to_vec
  (#et:Type0) {| sized et |}
  (#d0 #d1 #d2 : szp)
  (#l : layout d0 d1 d2 { is_full l })
  (a : vec et)
  (gm : t et l)
  (#s : erased (seq et){Seq.length s == d0 * d1 * d2})
  (#em : EMatrix3.t et d0 d1 d2)
  preserves
    cpu ** on gpu_loc (gm |-> em)
  requires
    a |-> s
  ensures
    a |-> to_seq l em
{
  map_loc gpu_loc
    #(gm |-> em)
    #(gm |-> em ** pure (SZ.fits (layout_size l)))
    fn () { pts_to_ref gm; };
  Pulse.Lib.Vec.pts_to_len a;
  map_loc gpu_loc
    #(gm |-> em)
    #(core gm |-> Frac 1.0R (to_seq l em))
    fn () { lower gm; };
  gpu_memcpy_device_to_host a (core gm) (d0 *^ (d1 *^ d2));
  map_loc gpu_loc
    #(core gm |-> Frac 1.0R (to_seq l em))
    #(gm |-> em)
    fn () {
      raise l (core gm);
      rewrite from_array l (core gm) |-> Frac 1.0R em
           as gm |-> em;
    };
}

ghost
fn share_n
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (k : pos)
  (#f : perm) (#s : EMatrix3.t et d0 d1 d2)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s
{
  unfold pts_to a #f s;
  T.tensor_share_n a k;
  forevery_map
    (fun (i:natlt k) -> T.tensor_pts_to a #(f /. k) s)
    (fun (i:natlt k) -> pts_to a #(f /. k) s)
    fn i { fold pts_to a #(f /. k) s };
}

ghost
fn gather_n
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (k : pos)
  (#f : perm) (#s : EMatrix3.t et d0 d1 d2)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s
{
  forevery_map
    (fun (i:natlt k) -> pts_to a #(f /. k) s)
    (fun (i:natlt k) -> T.tensor_pts_to a #(f /. k) s)
    fn i { unfold pts_to a #(f /. k) s };
  T.tensor_gather_n a k;
  fold pts_to a #f s;
}

inline_for_extraction noextract
fn read
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2) {| ctlayout l |}
  (a : t et l)
  (idx : raw_cit{cit_fits d0 d1 d2 idx})
  (#f : perm)
  (#s : erased (EMatrix3.t et d0 d1 d2))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == EMatrix3.macc s (pi_3_0 idx) (pi_3_1 idx) (pi_3_2 idx))
{
  unfold pts_to a #f s;
  let v = T.tensor_read a (adapt_cit_back d0 d1 d2 idx);
  fold pts_to a #f s;
  v
}

#push-options "--ifuel 3" // sigh
inline_for_extraction noextract
fn write
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2) {| ctlayout l |}
  (a : t et l)
  (idx : raw_cit{cit_fits d0 d1 d2 idx})
  (v : et)
  (#s : erased (EMatrix3.t et d0 d1 d2))
  requires
    a |-> s
  ensures
    a |-> EMatrix3.mupd s (pi_3_0 idx) (pi_3_1 idx) (pi_3_2 idx) v
{
  unfold pts_to a s;
  T.tensor_write a (adapt_cit_back d0 d1 d2 idx) v;
  fold pts_to a (EMatrix3.mupd s (pi_3_0 idx) (pi_3_1 idx) (pi_3_2 idx) v);
  ()
}
#pop-options

let pts_to_cell
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] ijk : ait d0 d1 d2)
  (v : et)
  : slprop
  = T.tensor_pts_to_cell a #f (adapt_idx_back ijk) v

let pts_to_cell_eq
  (#et : Type) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (ijk : ait d0 d1 d2) (f : perm) (v : et)
  : Lemma (pts_to_cell a #f ijk v
           ==
           B.pts_to_cell (core a) #f (l.imap.f (adapt_idx_back ijk)) v)
  = T.tensor_pts_to_cell_eq a (adapt_idx_back ijk) f v

ghost
fn explode
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix3.t et d0 d1 d2)
  requires a |-> Frac f s
  ensures
    forall+ (ijk : ait d0 d1 d2).
      Cell a ijk |-> Frac f (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk))
{
  unfold pts_to a #f s;
  T.tensor_explode a;
  forevery_iso abs_bij _;
  forevery_ext _ (fun (ijk : ait d0 d1 d2) -> Cell a ijk |-> Frac f (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk)));
  ()
}

ghost
fn implode
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix3.t et d0 d1 d2)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (ijk : ait d0 d1 d2).
      Cell a ijk |-> Frac f (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk))
  ensures
    a |-> Frac f s
{
  forevery_iso (bij_sym abs_bij) _;
  forevery_ext _ (fun (i : abs (desc d0 d1 d2)) -> Cell a i |-> Frac f (acc s i));
  T.tensor_implode a;
  fold pts_to a #f s;
}

ghost
fn ilower
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix3.t et d0 d1 d2)
  requires
    a |-> Frac f s
  ensures
    pure (SZ.fits (layout_size l)) **
    (forall+ (ijk : ait d0 d1 d2).
      pts_to_cell a #f ijk (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk)))
{
  pts_to_ref a;
  explode a;
  forevery_ext
    (fun (ijk : ait d0 d1 d2) ->
      Cell a ijk |-> Frac f (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk)))
    (fun (ijk : ait d0 d1 d2) ->
      pts_to_cell a #f ijk (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk)));
}

ghost
fn iraise
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l)
  (#f : perm)
  (#s : EMatrix3.t et d0 d1 d2)
  requires
    pure (SZ.fits (layout_size l)) **
    (forall+ (ijk : ait d0 d1 d2).
      pts_to_cell a #f ijk (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk)))
  ensures
    a |-> Frac f s
{
  forevery_ext
    (fun (ijk : ait d0 d1 d2) ->
      pts_to_cell a #f ijk (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk)))
    (fun (ijk : ait d0 d1 d2) ->
      Cell a ijk |-> Frac f (EMatrix3.macc s (pi_3_0 ijk) (pi_3_1 ijk) (pi_3_2 ijk)));
  implode a;
}

(* ---- page: extract a 2-D slice (Array2) from a 3-D tensor (Array3) ---- *)

inline_for_extraction noextract
let page
  (#et : Type0)
  (#d0 #d1 #d2 : erased nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  (i : erased nat{i < d0})
  : Array2.t et (page_layout a i)
  = T.sliceof a 0 i

let page_is_global
  (#et : Type0) (#d0 #d1 #d2 : nat) (#l : layout d0 d1 d2)
  (a : t et l) (i : erased nat{i < d0})
  : Lemma (ensures Array2.is_global (page a i) <==> is_global a)
          [SMTPat (page a i)]
  = lem_is_global_iff_sliceof a 0 i // weird

ghost
fn extract_page
  (#et : Type0)
  (#d0 #d1 #d2 : nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  (i : natlt d0)
  (#f : perm) (#s : EMatrix3.t et d0 d1 d2)
  requires
    a |-> Frac f s
  ensures
    page a i |-> Frac f (EMatrix3.slice_page s i) **
    (forall* (s' : EMatrix.ematrix et d1 d2).
      page a i |-> Frac f s' @==>
      a |-> Frac f (EMatrix3.upd_page s i s'))
{
  (* This whole thing could eventually work by unification. *)

  unfold pts_to a #f s;
  T.tensor_extract_slice a 0 i #f #s;

  rewrite each T.sliceof a 0 i as page a i;
  fold Array2.pts_to (page a i) #f (EMatrix3.slice_page s i);

  intro_forall
    #_
    #(fun (s' : EMatrix.ematrix et d1 d2) ->
      page a i |-> Frac f s'
      @==> a |-> Frac f (EMatrix3.upd_page s i s'))
    (forall* (s' : chest (modulo_i 0 (desc d0 d1 d2)) et).
      sliceof a 0 i |-> Frac f s'
      @==> a |-> Frac f (chest_update_slice 0 i s s'))
    fn s' {
      intro_trade
        (page a i |-> Frac f s')
        (a |-> Frac f (EMatrix3.upd_page s i s'))
        (forall* (s' : chest (modulo_i 0 (desc d0 d1 d2)) et).
              sliceof a 0 i |-> Frac f s'
              @==> a |-> Frac f (chest_update_slice 0 i s s'))
        fn _ {
          let w : chest (modulo_i 0 (desc d0 d1 d2)) et = s';
          elim_forall w;
          rewrite Array2.pts_to (page a i) #f s'
               as sliceof a 0 i |-> Frac f w;
          elim_trade _ _;
        };
    };
}

ghost
fn extract_page_ro
  (#et : Type0)
  (#d0 #d1 #d2 : nat)
  (#l : layout d0 d1 d2)
  (a : t et l)
  (i : natlt d0)
  (#f : perm) (#s : EMatrix3.t et d0 d1 d2)
  requires
    a |-> Frac f s
  ensures
    factored
      (page a i |-> Frac f (EMatrix3.slice_page s i))
      (a |-> Frac f s)
{
  extract_page a i;
  elim_forall (EMatrix3.slice_page s i);
  assert pure (EMatrix3.equal (EMatrix3.upd_page s i (EMatrix3.slice_page s i)) s);
}