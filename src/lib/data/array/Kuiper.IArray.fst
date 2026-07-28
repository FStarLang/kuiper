module Kuiper.IArray
#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.IView
module T = FStar.Tactics.V2
module SZ = Kuiper.SizeT
module B = Kuiper.Array
module Trade = Pulse.Lib.Trade

inline_for_extraction
type iarray (et : Type0) (vw : aiview) : Type0 =
  larray et (len vw)

let is_global (#et : Type0) (#vw : aiview) (arr : iarray et vw) : prop =
  B.is_global_array arr

inline_for_extraction noextract
let from_array
  (#et : Type0)
  (vw : aiview)
  (arr : larray et (len vw))
  : iarray et vw
  = arr

let core a = a

let lem_is_global_iff_core
  (#a : Type0)
  (#vw : aiview)
  (g : iarray a vw)
  : Lemma (ensures is_global g <==> is_global_array (core g))
          [SMTPat (is_global g)]
  = ()

let lem_from_array_core
  (#et : Type0)
  (#vw : aiview)
  (arr : iarray et vw)
  : Lemma (ensures from_array vw (core arr) == arr)
          [SMTPat (core arr)]
  = ()

let lem_core_from_array
  (#et : Type0)
  (vw : aiview)
  (p : larray et (len vw))
  : Lemma (ensures core (from_array vw p) == p)
          [SMTPat (from_array vw p)]
  = ()

(* Ownership over a single index. *)
let iarray_pts_to_cell
  (#et:Type0)
  (#vw : aiview)
  ([@@@mkey] a : iarray et vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.ait)
  (v : et)
  : slprop
  = pts_to_cell (core a) #f (it_to_nat vw i) v

let iarray_pts_to_cell_def
  (#et : Type)
  (#vw : aiview)
  (a : iarray et vw)
  (#f : perm)
  (i : vw.ait)
  (v : et)
  : Lemma (iarray_pts_to_cell a #f i v ==
            pts_to_cell (core a) #f (it_to_nat vw i) v)
  = ()

let iarray_pts_to
  (#et:Type0) (#vw : aiview)
  ([@@@mkey] a : iarray et vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : (vw.ait -> GTot et))
  : slprop
  = pure (SZ.fits (len vw)) **
    (forall+ (i : vw.ait).
      iarray_pts_to_cell a #f i (v i))

let is_send_across_iarray
  (#et : Type0)
  (#vw : aiview)
  (x : iarray et vw)
  (vis : visibility)
  (#_ : squash (visibility_of (core x) == vis))
  (#f : perm)
  (v : vw.ait -> GTot et)
  : is_send_across vis (iarray_pts_to x #f v)
=
  let i :
    is_send_across (visibility_of (core x)) (iarray_pts_to x #f v) =
    solve in
  i

instance is_send_across_global_iarray
  (#et:Type0)
  (#vw : aiview)
  (x: iarray et vw { is_global x })
  (#f : perm)
  (v : (vw.ait -> GTot et))
  : is_send_across gpu_of (iarray_pts_to x #f v)
  = Tactics.Typeclasses.solve

instance is_send_across_global_iarray_cell
  (#et:Type0)
  (#vw : aiview)
  (a : iarray et vw { is_global a })
  (#f : perm)
  (i : vw.ait)
  (v : et)
  : is_send_across gpu_of (iarray_pts_to_cell a #f i v)
  = solve

ghost
fn iarray_pts_to_ref
  (#et:Type0)
  (#vw : aiview )
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.ait -> GTot et))
  preserves
    a |-> Frac f v
  ensures
    pure (SZ.fits (len vw))
{
  unfold iarray_pts_to a #f v;
  fold iarray_pts_to a #f v;
}

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
{
  unfold iarray_pts_to a #f v1;
  forevery_ext
    (fun i -> iarray_pts_to_cell a #f i (v1 i))
    (fun i -> iarray_pts_to_cell a #f i (v2 i));
  fold iarray_pts_to a #f v2;
}

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
{
  unfold iarray_pts_to a #f v;
}

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
{
  fold iarray_pts_to a #f v;
}

(* Begin viewing something abstractly, with the trivial view. *)
ghost
fn iarray_begin_
  (#et:Type0)
  (#len : erased nat)
  (a : larray et len)
  (#f : perm)
  (#v : lseq et len)
  requires
    a |-> Frac f v
  ensures
    from_array (raw_view #len) a |-> Frac f (g_seq_acc v)
{
  B.array_slice_1 a;
  fold iarray_pts_to (from_array (raw_view #len) a) #f (fun i -> v @! i);
}

fn iarray_begin
  (#et:Type0)
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
{
  iarray_begin_ a;
  from_array (raw_view #len) a;
}

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
{
  unfold iarray_pts_to a #f v;
  forevery_ext
    (fun (i : natlt len) ->
      iarray_pts_to_cell a #f i (v i))
    (fun (i : natlt len) ->
      iarray_pts_to_cell a #f i (Seq.init_ghost len v @! i));
  B.array_unslice_1 (core a);
}

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
{
  iarray_end_ a;
  core a;
}

ghost
fn iarray_end2_
  (#et : Type0)
  (#vw : aiview { is_full_view vw })
  (a : iarray et vw)
  (#f : perm)
  (#v : vw.ait -> GTot et)
  requires
    a |-> Frac f v
  ensures
    core a |-> Frac f (Seq.init_ghost (len vw) (fun (i : natlt (len vw)) -> v (it_of_nat vw i)))
{
  unfold iarray_pts_to a #f v;
  let b : (vw.ait =~ natlt vw.len) = full_view_bij vw;
  forevery_iso b _;

  let s = Seq.init_ghost (len vw) (fun (i : natlt (len vw)) -> v (it_of_nat vw i));
  forevery_ext #(natlt vw.len)
    (fun i -> iarray_pts_to_cell a #f (b.gg i) (v (b.gg i)))
    (fun i -> pts_to_cell (core a) #f i (s @! i));

  B.array_unslice_1 (core a);
}

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
{
  iarray_end2_ a;
  core a;
}

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
{
  rewrite
    Cell a i |-> Frac f v
  as
    Cell a' i' |-> Frac f v;
}

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
{
  unfold iarray_pts_to a #f v;
  forevery_iso bij _;
  forevery_ext
    (fun i -> iarray_pts_to_cell a #f (bij.gg i) (v (bij.gg i)))
    (fun i -> iarray_pts_to_cell (from_array (reindex_view vw bij) (core a)) #f i (v (bij.gg i)));
  forevery_rw_type ait' (reindex_view vw bij).ait _;
  fold iarray_pts_to
    (from_array (reindex_view vw bij) (core a)) #f
    (fun i -> v (bij.gg i));
  ()
}

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
{
  iarray_reindex_ bij a;
  (from_array (reindex_view vw bij) (core a));
}

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
{
  unfold iarray_pts_to a #f v;
  forevery_split_either #vw1.ait #vw2.ait _;

  forevery_ext #vw1.ait
    (fun i -> iarray_pts_to_cell a #f (Inl i) (v (Inl i)))
    (fun i -> iarray_pts_to_cell (from_array vw1 (core a)) #f i (v (Inl #vw1.ait #vw2.ait i)));
  fold iarray_pts_to (from_array vw1 (core a)) #f (fun i -> v (Inl i));

  forevery_ext #vw2.ait
    (fun i -> iarray_pts_to_cell a #f (Inr i) (v (Inr i)))
    (fun i -> iarray_pts_to_cell (from_array vw2 (core a)) #f i (v (Inr #vw1.ait #vw2.ait i)));
  fold iarray_pts_to (from_array vw2 (core a)) #f (fun i -> v (Inr i));
}

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
{
  iarray_split2_ vw1 vw2 a;
  (from_array vw1 (core a), from_array vw2 (core a));
}

ghost
fn iarray_split_n
  (#et : Type0)
  (#n : pos)
  (vws : natlt n -> aiview { forall i. len (vws i) = len (vws 0) })
  (#_ : squash (no_overlap_fam n vws))
  (a : iarray et (sum_aiview_fam n vws #()))
  (#f : perm)
  (#v : (i:natlt n & (vws i).ait) -> GTot et)
  requires
    a |-> Frac f v
  ensures
    forall+ (i : natlt n).
      from_array (vws i) (core a) |-> Frac f (fun j -> v (| i, j |))
{
  unfold iarray_pts_to a #f v;
  forevery_rw_type (sum_aiview_fam n vws).ait (i : natlt n & (vws i).ait) _;
  forevery_unflatten_dep' #(natlt n) #(fun i -> (vws i).ait) _;
  ghost
  fn aux (i : natlt n)
    requires
      forall+ (j : (vws i).ait).
        iarray_pts_to_cell a #f (| i, j |) (v (| i, j |))
    ensures
      from_array (vws i) (core a) |-> Frac f (fun j -> v (| i, j |))
  {
    ghost
    fn aux2 (j : (vws i).ait)
      requires
        iarray_pts_to_cell a #f (| i, j |) (v (| i, j |))
      ensures
        iarray_pts_to_cell (from_array (vws i) (core a)) #f j (v (| i, j |))
    {
      unfold iarray_pts_to_cell a #f (| i, j |) (v (| i, j |));
      fold iarray_pts_to_cell (from_array (vws i) (core a)) #f j (v (| i, j |))
    };
    forevery_map _ _ aux2;
    fold iarray_pts_to (from_array (vws i) (core a)) #f (fun j -> v (| i, j |));
    ()
  };
  forevery_map _ _ aux;
  ();
}

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
{
  (* Boring: share everything N-wise under the forall+, then commute
  the two forall+ *)
  unfold iarray_pts_to a #f v;
  forevery_map
    (fun i -> pts_to_slice (core a) #f (it_to_nat vw i) (it_to_nat vw i + 1) seq![v i])
    (fun i -> forall+ (_:natlt k). pts_to_slice (core a) #(f /. Real.of_int k) (it_to_nat vw i) (it_to_nat vw i + 1) seq![v i])
    fn i { slice_share (core a) _ _ k };

  forevery_commute _;
  forevery_map #(natlt k)
    (fun _ -> forall+ (x:vw.ait).
      pts_to_slice (core a) #(f /. Real.of_int k) (it_to_nat vw x) (it_to_nat vw x + 1) seq![v x])
    (fun _ -> a |-> Frac (f /. Real.of_int k) v)
    fn _ {
      forevery_ext #(vw.ait)
        (fun i -> pts_to_cell (core a) #(f /. Real.of_int k) (it_to_nat vw i) (v i))
        (fun i -> iarray_pts_to_cell a #(f /. Real.of_int k) i (v i));
      fold iarray_pts_to a #(f /. Real.of_int k) v;
    };
}

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
{
  (* Grab one out to get the pure fact about the length. *)
  forevery_natlt_pop k _;
  unfold iarray_pts_to a #(f /. Real.of_int k) v;
  fold   iarray_pts_to a #(f /. Real.of_int k) v;
  forevery_natlt_push k _;
  assert pure (SZ.fits (len vw));

  forevery_map #(natlt k)
    (fun _ -> a |-> Frac (f /. Real.of_int k) v)
    (fun _ -> forall+ (x:vw.ait).
      pts_to_slice (core a) #(f /. Real.of_int k) (it_to_nat vw x) (it_to_nat vw x + 1) seq![v x])
    fn _ {
      unfold iarray_pts_to a #(f /. Real.of_int k) v;
    };
  forevery_commute _;
  forevery_map
    (fun i -> forall+ (_ : natlt k). iarray_pts_to_cell a #(f /. Real.of_int k) i (v i))
    (fun i -> iarray_pts_to_cell a #f i (v i))
    fn i {
      slice_gather (core a) _ _ k;
      fold iarray_pts_to_cell a #f i (v i);
    };
  fold iarray_pts_to a #f v;
}

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
{
  unfold iarray_pts_to a #f1 v1;
  unfold iarray_pts_to a #f2 v2;
  forevery_zip (fun i -> iarray_pts_to_cell a #f1 i (v1 i))
               (fun i -> iarray_pts_to_cell a #f2 i (v2 i));
  ghost
  fn aux (i : vw.ait)
    requires
      iarray_pts_to_cell a #f1 i (v1 i) **
      iarray_pts_to_cell a #f2 i (v2 i)
    ensures
      iarray_pts_to_cell a #f1 i (v2 i) **
      iarray_pts_to_cell a #f2 i (v2 i)
  {
    unfold iarray_pts_to_cell a #f1 i (v1 i);
    unfold iarray_pts_to_cell a #f2 i (v2 i);
    slice_pts_to_eq (core a) (it_to_nat vw i) (it_to_nat vw i + 1) #f1 f2;
    assert pure (seq![v1 i] == seq![v2 i]);
    assert pure (seq![v1 i] @! 0 == seq![v2 i] @! 0); // sad...
    assert pure (v1 i == v2 i);
    fold iarray_pts_to_cell a #f1 i (v1 i);
    fold iarray_pts_to_cell a #f2 i (v2 i);
  };
  forevery_map #(vw.ait) _ _ aux;
  forevery_unzip _ _;
  fold iarray_pts_to a #f1 v2;
  fold iarray_pts_to a #f2 v2;
}

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
{
  let ai : erased vw.ait = ci |> cw.sch.bij.gg; (* abstract index *)
  let ni = cw.step.cimap.cf ci;                    (* numerical index *)
  rewrite each ci_to_ai vw ci as ai;

  cw.step.compat ai;
  assert pure ((cw.step.cimap.cf (cw.sch.bij.ff ai)) == SZ.uint_to_t (vw.step.imap.f ai));
  assert pure (SZ.v (cw.step.cimap.cf (cw.sch.bij.ff ai)) == vw.step.imap.f ai);

  unfold iarray_pts_to_cell a ai v0;
  rewrite each it_to_nat vw (reveal ai) as ni;
  slice_write (core a) ni v1;
  with s'. assert (B.pts_to_slice (core a) ni (ni+1) s');
  Kuiper.Seq.Common.lem_one_elem s' v1;
  rewrite each SZ.v ni as (ci_to_ai vw ci |~> vw.step.imap);
  with i v. rewrite
    B.pts_to_slice (core a) i (i+1) v
  as
    B.pts_to_slice (core a) i (i+1) seq![v @! 0];
  fold iarray_pts_to_cell a (ci_to_ai vw ci) v1;
  ();
}

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
{
  rewrite each ai as ci_to_ai vw ci;
  let res = iarray_write_cell #et  a ci v1;
  rewrite each ci_to_ai vw ci as ai;
  res
}

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
{
  let ai : erased vw.ait = ci |> cw.sch.bij.gg; (* abstract index *)
  let ni = cw.step.cimap.cf ci;                    (* numerical index *)
  rewrite each ci_to_ai vw ci as ai;

  cw.step.compat ai;
  assert pure ((cw.step.cimap.cf (cw.sch.bij.ff ai)) == SZ.uint_to_t (vw.step.imap.f ai));
  (* ^ FIXME: this should be exactly what we get from the line above? *)
  assert pure (SZ.v (cw.step.cimap.cf (cw.sch.bij.ff ai)) == vw.step.imap.f ai);

  unfold iarray_pts_to_cell a #f ai v0;
  rewrite each it_to_nat vw (reveal ai) as ni;
  let res = B.slice_read (core a) ni;
  with s'. assert (B.pts_to_slice (core a) #f ni (ni+1) s');
  rewrite each SZ.v ni as (ci_to_ai vw ci |~> vw.step.imap);
  fold iarray_pts_to_cell a #f (ci_to_ai vw ci) v0;
  res;
}

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
{
  rewrite each ai as ci_to_ai vw i;
  let res = iarray_read_cell #et a i;
  rewrite each ci_to_ai vw i as ai;
  res
}

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
{
  unfold iarray_pts_to a #f v;
  forevery_extract (ci_to_ai vw ci) _;
  let res = iarray_read_cell a ci #f;
  Trade.elim_trade _ _;
  fold iarray_pts_to a #f v;
  res
}

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
    (a |-> oplus v0 (ci_to_ai vw ci) e)
{
  unfold iarray_pts_to a v0;
  forevery_extract_if (ci_to_ai vw ci) _;
  iarray_write_cell a ci e;

  forevery_intro_if (ci_to_ai vw ci) (fun i -> iarray_pts_to_cell a i e);
  forevery_zip #vw.ait
    (fun i -> if t2b (i == ci_to_ai vw ci)
              then iarray_pts_to_cell a i e
              else emp)
    _;
  ghost
  fn aux (i : vw.ait)
    requires
      (if t2b (i == ci_to_ai vw ci)
       then iarray_pts_to_cell a i e
       else emp)
    **
      (if t2b (i == ci_to_ai vw ci)
       then emp
       else iarray_pts_to_cell a i (v0 i))
    ensures
       iarray_pts_to_cell a i (oplus v0 (ci_to_ai vw ci) e i)
  {
    let cond = t2b (i == ci_to_ai vw ci);
    if cond {
      rewrite each (ci_to_ai vw ci) as i;
      rewrite each t2b True as true;
      assert (pure (e == oplus v0 i e i));
    } else {
      rewrite each (t2b (i == ci_to_ai vw ci)) as false;
      ();
    };
  };
  forevery_map _ _ aux;
  fold iarray_pts_to a (oplus v0 (ci_to_ai vw ci) e);
  ()
}
