module Kuiper.Tensor
#lang-pulse

open Kuiper
open Kuiper.Shape
open Kuiper.Bijection
module A = Kuiper.VArray
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2

let tensor (et : Type0) (#r : nat) (#d : shape r) (l : tlayout d) : Type0 =
  A.varray (tensor_aview et l)

let is_global
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l) : prop
  = A.is_global a

inline_for_extraction noextract
let from_array
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (l : tlayout d)
  (a : larray et (tlayout_ulen l))
  : tensor et l
  = A.from_array (tensor_aview et l) a

inline_for_extraction noextract
let core
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  : larray et (tlayout_ulen l)
  = A.core a

let lem_core_from_array
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]
  = ()

let lem_from_array_core
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (p : larray et (tlayout_ulen l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let lem_is_global_iff_core
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]
  = ()

let tensor_pts_to
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  ([@@@mkey] a : tensor et l)
  (#[T.exact (`1.0R)] f : perm)
  (s : chest d et)
  : slprop
  = A.varray_pts_to a #f s

let is_send_across_tensor
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (vis : visibility)
  (#_ : squash (visibility_of (core a) == vis))
  (#f : perm)
  (s : chest d et)
  : is_send_across vis (tensor_pts_to a #f s)
= A.is_send_across_varray a vis #_ #f s

instance is_send_across_global_tensor
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l { is_global a })
  (#f : perm) (s : chest d et)
  : is_send_across gpu_of (tensor_pts_to a #f s)
  = solve

(* This is slightly odd, the user needs to give the total size
instead of each dimension. *)
inline_for_extraction noextract
fn alloc0
  (#et:Type) {| sized et |}
  (#r : nat) (#d : shape r)
  (s : szp{SZ.v s == sizeof d})
  (l : tlayout d { is_full l })
  preserves
    cpu
  returns
    p : tensor et l
  ensures
    exists* em. on gpu_loc (p |-> em)
  ensures
    pure (is_global p) **
    pure (is_full_array (core p))
{
  let t = A.varray_alloc0 #et s (tensor_aview et l);
  with em. assert on gpu_loc (A.varray_pts_to t em);
  rewrite on gpu_loc (A.varray_pts_to t em)
       as on gpu_loc (tensor_pts_to t em);
  t
}

inline_for_extraction noextract
fn free
  (#et:Type)
  (#r : nat) (#d : shape r)
  (#l : tlayout d { is_full l })
  (p : tensor et l)
  (#em : chest d et)
  preserves
    cpu
  requires
    pure (is_full_array (core p)) **
    on gpu_loc (p |-> em)
  ensures emp
{
  rewrite on gpu_loc (tensor_pts_to p em)
       as on gpu_loc (A.varray_pts_to p  em);
  A.varray_free p;
}

ghost
fn tensor_pts_to_ref
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm) (#s : chest d et)
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_ulen l))
{
  unfold tensor_pts_to a #f s;
  A.varray_pts_to_ref a;
  fold tensor_pts_to a #f s;
}

ghost
fn tensor_pts_to_ref_located
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (#loc : loc_id)
  (#f : perm) (#s : chest d et)
  preserves
    on loc (a |-> Frac f s)
  ensures
    pure (SZ.fits (tlayout_ulen l))
{
  map_loc loc
    #(a |-> Frac f s)
    #(a |-> Frac f s ** pure (SZ.fits (tlayout_ulen l)))
  fn _ {
    tensor_pts_to_ref a;
  };
}

ghost
fn tensor_pts_to_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f1 f2 : perm)
  (#s1 #s2 : chest d et)
  requires
    tensor_pts_to a #f1 s1 **
    tensor_pts_to a #f2 s2
  ensures
    tensor_pts_to a #f1 s2 **
    tensor_pts_to a #f2 s2
{
  unfold tensor_pts_to a #f1 s1;
  unfold tensor_pts_to a #f2 s2;
  A.varray_pts_to_eq a f2;
  fold tensor_pts_to a #f1 s2;
  fold tensor_pts_to a #f2 s2;
}

ghost
fn tensor_concr
  (#et:Type)
  (#r : nat) (#d : shape r)
  (#l : tlayout d { is_full l })
  (g : tensor et l)
  (#s : chest d et)
  (#f : perm)
  requires
    g |-> Frac f s
  ensures
    core g |-> Frac f (to_seq l s)
{
  unfold tensor_pts_to g #f s;
  A.varray_concr g;
  to_seq_rel l s;
  rewrite A.core g |-> Frac f (A.to_seq (tensor_aview et l) s)
       as core g |-> Frac f (to_seq l s);
}

ghost
fn tensor_abs
  (#et:Type)
  (#r : nat) (#d : shape r)
  (l : tlayout d { is_full l })
  (p : larray et (tlayout_ulen l))
  (#f : perm)
  (#s : chest d et)
  requires
    p |-> Frac f (to_seq l s)
  ensures
    from_array l p |-> Frac f s
{
  to_seq_rel l s;
  rewrite
    p |-> Frac f (to_seq l s)
  as
    p |-> Frac f (A.to_seq (tensor_aview et l) s);
  A.varray_abs (tensor_aview et l) p;
  fold tensor_pts_to (from_array l p) #f s;
}

ghost
fn tensor_abs'
  (#et:Type)
  (#r : nat) (#d : shape r)
  (l : tlayout d { is_full l })
  (p : larray et (tlayout_ulen l))
  (#f : perm)
  (#s : lseq et (tlayout_ulen l))
  requires
    p |-> Frac f s
  ensures
    from_array l p |-> Frac f (from_seq l s)
{
  rewrite each s as to_seq l (from_seq l s);
  tensor_abs l p;
}

ghost
fn tensor_share_n
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l) (k : pos)
  (#f : perm) (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s
{
  unfold tensor_pts_to a #f s;
  A.varray_share_n a k;
  forevery_map
    (fun (i:natlt k) -> A.varray_pts_to a #(f /. k) s)
    (fun (i:natlt k) -> tensor_pts_to a #(f /. k) s)
    fn i { fold tensor_pts_to a #(f /. k) s };
}

ghost
fn tensor_gather_n
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l) (k : pos)
  (#f : perm) (#s : chest d et)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s
{
  forevery_map
    (fun (i:natlt k) -> tensor_pts_to a #(f /. k) s)
    (fun (i:natlt k) -> A.varray_pts_to a #(f /. k) s)
    fn i { unfold tensor_pts_to a #(f /. k) s };
  A.varray_gather_n a k;
  fold tensor_pts_to a #f s;
}

ghost
fn tensor_gather_n_underspec
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l) (k : pos)
  (#f : perm)
  requires
    forall+ (_:natlt k).
      exists* (s : chest d et). tensor_pts_to a #(f /. k) s
  ensures
    exists* (s : chest d et). tensor_pts_to a #f s
{
  forevery_natlt_pop k _;
  with s. assert tensor_pts_to a #(f /. k) s;
  ghost
  fn aux (_ : natlt (k-1))
    norewrite
    requires
      tensor_pts_to a #(f /. k) s ** (exists* v. tensor_pts_to a #(f /. k) v)
    ensures
      tensor_pts_to a #(f /. k) s ** tensor_pts_to a #(f /. k) s
  {
    tensor_pts_to_eq a (f /. k) #_ #s;
  };
  forevery_map_extra #(natlt (k-1)) (tensor_pts_to a #(f /. k) s)
    (fun (_ : natlt (k-1)) -> exists* v. tensor_pts_to a #(f /. k) v)
    (fun (_ : natlt (k-1)) -> tensor_pts_to a #(f /. k) s)
    aux;
  forevery_natlt_push k _;
  tensor_gather_n a k;
}

inline_for_extraction noextract
fn tensor_read
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (i : conc d)
  (#f : perm)
  (#s : chest d et)
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == acc s (up i))
{
  unfold tensor_pts_to a #f s;
  let v = A.varray_read a i;
  fold tensor_pts_to a #f s;
  v
}

inline_for_extraction noextract
fn tensor_write
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (i : conc d)
  (v : et)
  (#s : chest d et)
  requires
    a |-> s
  ensures
    a |-> upd s (up i) v
{
  unfold tensor_pts_to a s;
  A.varray_write a i v;
  fold tensor_pts_to a;
}

let tensor_pts_to_cell
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  ([@@@mkey] a : tensor et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] i : abs d)
  (v : et)
  : slprop
  = A.varray_pts_to_cell a #f i v

let tensor_pts_to_cell_eq
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l) (i : abs d) (f : perm) (v : et)
  : Lemma (Cell a i |-> Frac f v
           ==
           pts_to_cell (core a) #f (l.imap.f i) v)
  = A.varray_pts_to_cell_eq a i f v

instance is_send_across_global_tensor_cell
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l { is_global a })
  (#f : perm) (i : abs d) (v : et)
  : is_send_across gpu_of (tensor_pts_to_cell a #f i v)
  = solve

ghost
fn tensor_explode
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    forall+ (i : abs d).
      Cell a i |-> Frac f (acc s i)

{
  unfold tensor_pts_to a #f s;
  A.varray_explode a;

  forevery_rw_type _ (abs d) _;
  forevery_ext
    (fun (i : abs d) ->
      Cell a i |-> Frac f ((tensor_aview et l).ctn.acc s i))
    (fun (i : abs d) ->
      Cell a i |-> Frac f (acc s i));
  ();
}

ghost
fn tensor_implode
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    pure (SZ.fits (tlayout_ulen l))
  requires
    forall+ (i : abs d).
      Cell a i |-> Frac f (acc s i)
  ensures
    a |-> Frac f s
{
  forevery_ext
    (fun (i : abs d) ->
      Cell a i |-> Frac f (acc s i))
    (fun (i : abs d) ->
      Cell a i |-> Frac f ((tensor_aview et l).ctn.acc s i));
  forevery_rw_type _ (tensor_aview et l).iview.ait _;
  A.varray_implode a;
  fold tensor_pts_to a #f s;
}

ghost
fn tensor_ilower
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i : abs d).
      pts_to_cell (core a) #f (l.imap.f i) (acc s i))
{
  tensor_pts_to_ref a;
  tensor_explode a;
  forevery_map
    (fun (i : abs d) -> Cell a i |-> Frac f (acc s i))
    (fun (i : abs d) -> pts_to_cell (core a) #f (l.imap.f i) (acc s i))
    fn i {
      tensor_pts_to_cell_eq a i f (acc s i);
      rewrite
        Cell a i |-> Frac f (acc s i)
      as
        pts_to_cell (core a) #f (l.imap.f i) (acc s i);
    };
}

ghost
fn tensor_iraise
  (#et : Type0) (#r : nat) (#d : shape r)
  (#l : tlayout d)
  (a : tensor et l)
  (#f : perm)
  (#s : chest d et)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (i : abs d).
      pts_to_cell (core a) #f (l.imap.f i) (acc s i))
  ensures
    a |-> Frac f s
{
  forevery_map
    (fun (i : abs d) -> pts_to_cell (core a) #f (l.imap.f i) (acc s i))
    (fun (i : abs d) -> Cell a i |-> Frac f (acc s i))
    fn i {
      tensor_pts_to_cell_eq a i f (acc s i);
      rewrite
        pts_to_cell (core a) #f (l.imap.f i) (acc s i)
      as
        Cell a i |-> Frac f (acc s i);
    };
  tensor_implode a;
}

inline_for_extraction noextract
fn tensor_read_cell
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (i : conc d)
  (#f : perm)
  (#s : erased et)
  preserves
    Cell a (up i) |-> Frac f s
  returns
    v : et
  ensures
    pure (v == s)
{
  unfold tensor_pts_to_cell a #f (up i) s;
  let v = A.varray_read_cell a i;
  fold tensor_pts_to_cell a #f (up i) s;
  v
}

inline_for_extraction noextract
fn tensor_write_cell
  (#et : Type0) (#r : erased nat) (#d : shape r)
  (#l : tlayout d) {| ctlayout l |}
  (a : tensor et l)
  (i : conc d)
  (v : et)
  (#s : erased et)
  requires
    Cell a (up i) |-> s
  ensures
    Cell a (up i) |-> v
{
  unfold tensor_pts_to_cell a (up i) s;
  A.varray_write_cell a i v;
  fold tensor_pts_to_cell a (up i) v;
  ()
}

(* Rank-2 conveniences: explode/implode/ilower/iraise presented over the
   (natlt rows & natlt cols) index pair, as special cases of the generic
   rank-r operations above. *)

inline_for_extraction noextract
unfold
let abs_bij2 (#rows #cols : nat)
  : (abs (rows @| cols @| INil) =~ (natlt rows & natlt cols)) =
  {
    ff = (fun (i, (j, ())) -> (i, j));
    gg = (fun (i, j) -> (i, (j, ())));
  }

ghost
fn tensor_explode2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : tensor et l)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    a |-> Frac f s
  ensures
    forall+ (ij : natlt rows & natlt cols).
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij)))
{
  tensor_explode a;
  forevery_iso #(abs (rows @| cols @| INil)) #(natlt rows & natlt cols)
    abs_bij2 (fun (i : abs (rows @| cols @| INil)) -> Cell a i |-> Frac f (acc s i));
  forevery_ext
    (fun (ij : natlt rows & natlt cols) ->
      Cell a (abs_bij2.gg ij) |-> Frac f (acc s (abs_bij2.gg ij)))
    (fun (ij : natlt rows & natlt cols) ->
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij))));
}

ghost
fn tensor_implode2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : tensor et l)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    pure (SZ.fits (tlayout_ulen l))
  requires
    forall+ (ij : natlt rows & natlt cols).
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij)))
  ensures
    a |-> Frac f s
{
  forevery_iso #(natlt rows & natlt cols) #(abs (rows @| cols @| INil))
    (bij_sym abs_bij2)
    (fun (ij : natlt rows & natlt cols) ->
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij))));
  forevery_ext
    (fun (i : abs (rows @| cols @| INil)) ->
      Cell a (idx2 (fst ((bij_sym abs_bij2).gg i)) (snd ((bij_sym abs_bij2).gg i)))
        |-> Frac f (acc s (idx2 (fst ((bij_sym abs_bij2).gg i)) (snd ((bij_sym abs_bij2).gg i)))))
    (fun (i : abs (rows @| cols @| INil)) -> Cell a i |-> Frac f (acc s i));
  tensor_implode a;
}

ghost
fn tensor_ilower2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : tensor et l)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    a |-> Frac f s
  ensures
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      Cell a (idx2 r c) |-> Frac f (acc s (idx2 r c)))
{
  tensor_pts_to_ref a;
  tensor_explode2 a;
  forevery_unflatten'
    (fun (ij : natlt rows & natlt cols) ->
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij))));
}

ghost
fn tensor_iraise2
  (#et : Type0) (#rows #cols : nat) (#l : layout2 rows cols)
  (a : tensor et l)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    pure (SZ.fits (tlayout_ulen l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      Cell a (idx2 r c) |-> Frac f (acc s (idx2 r c)))
  ensures
    a |-> Frac f s
{
  forevery_flatten'
    (fun (ij : natlt rows & natlt cols) ->
      Cell a (idx2 (fst ij) (snd ij)) |-> Frac f (acc s (idx2 (fst ij) (snd ij))));
  tensor_implode2 a;
}

let ref_of_tensor_cell
  (#et : Type0)
  (#r : nat) (#s : shape r) (#l : tlayout s)
  (a : tensor et l)
  (i : abs s)
  : GTot (ref et)
  = Array.Core.ref_of_array_cell (core a) (l.imap.f i)

inline_for_extraction noextract
fn get_ref_of_tensor_cell
  (#et : Type0)
  (#r : nat) (#s : shape r) (#l : tlayout s)
  (a : tensor et l) {| c : ctlayout l |}
  (i : conc s)
  returns
    r : ref et
  ensures
    pure (r == ref_of_tensor_cell a (up i))
{
  Array.Core.get_ref_of_array_cell (core a) (c.cimap i)
}

ghost
fn tensor_cell_to_ref
  (#et : Type0)
  (#r : nat) (#s : shape r) (#l : tlayout s)
  (a : tensor et l)
  (i : abs s)
  (#f : perm)
  (#v : erased et)
  requires
    Cell a i |-> Frac f v
  ensures
    ref_of_tensor_cell a i |-> Frac f v
{
  tensor_pts_to_cell_eq a i f v;
  rewrite Cell a i |-> Frac f v
       as pts_to_cell (core a) #f (l.imap.f i) v;
  Array.Core.array_cell_to_ref (core a) (l.imap.f i);
}

ghost
fn tensor_cell_from_ref
  (#et : Type0)
  (#r : nat) (#s : shape r) (#l : tlayout s)
  (a : tensor et l)
  (i : abs s)
  (#f : perm)
  (#v : erased et)
  requires
    ref_of_tensor_cell a i |-> Frac f v
  ensures
    Cell a i |-> Frac f v
{
  tensor_pts_to_cell_eq a i f v;
  Array.Core.array_cell_from_ref (core a) (l.imap.f i);
  rewrite pts_to_cell (core a) #f (l.imap.f i) v
       as Cell a i |-> Frac f v;
}
