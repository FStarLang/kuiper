module Kuiper.IArray
#lang-pulse

open Kuiper
open Kuiper.Injection
open Kuiper.IView
module T = FStar.Tactics.V2
module SZ = FStar.SizeT
module B = Kuiper.Array
module Trade = Pulse.Lib.Trade

noeq
inline_for_extraction
type iarray (et : Type0) (vw : aiview) : Type0 =
  | IA of B.gpu_array et (len vw)

inline_for_extraction noextract
let from_array
  (#et : Type0)
  (vw : aiview)
  (arr : gpu_array et (len vw))
  : iarray et vw
  = IA arr

let core (IA a) = a

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
  (p : gpu_array et (len vw))
  : Lemma (ensures core (from_array vw p) == p)
          [SMTPat (from_array vw p)]
  = ()

(* Ownership over a single index. *)
let iarray_pts_to_cell
  (#et:Type0)
  (#vw : aiview)
  ([@@@mkey] a : iarray et vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : vw.sch.ait)
  (v : et)
  : slprop
  = gpu_pts_to_cell (core a) #f (i |~> vw.step.imap) seq![v]

let iarray_pts_to
  (#et:Type0) (#vw : aiview)
  ([@@@mkey] a : iarray et vw)
  (#[T.exact (`1.0R)] f : perm)
  (v : (vw.sch.ait -> GTot et))
  : slprop
  = pure (SZ.fits (len vw)) **
    (forall+ (i : vw.sch.ait).
      iarray_pts_to_cell a #f i (v i))

ghost
fn iarray_pts_to_ref
  (#et:Type0)
  (#vw : aiview )
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.sch.ait -> GTot et))
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
  (v1 v2 : (vw.sch.ait -> GTot et))
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

(* Note: the functions below use the Enumerable instance for vw.sch.ait
   that is inside the aview record.
   We do this since it's
   not necessary for that enumeration to match the one in the typeclass system.
   For example, for a matrix view, that enumeration can be anything
   depending on the layout chosen, but the enumeration we want for the
   **abstract indices** is just lexicographic. *)

ghost
fn iarray_explode
  (#et:Type)
  (#vw : aiview)
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.sch.ait -> GTot et))
  requires
    a |-> Frac f v
  ensures
    forall+ (i : vw.sch.ait).
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
  (#v : (vw.sch.ait -> GTot et))
  requires pure (SZ.fits (len vw))
  requires
    forall+ (i : vw.sch.ait).
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
  (a : gpu_array et len)
  (#f : perm)
  (#v : lseq et len)
  requires
    a |-> Frac f v
  ensures
    from_array (raw_view #len) a |-> Frac f (g_seq_acc v)
{
  B.gpu_pts_to_slice_ref a 0 len;
  B.gpu_array_slice_1 a;
  (* WOW! *)
  forevery_fromstar
    (fun (i : natlt len) ->
      iarray_pts_to_cell (from_array (raw_view #len) a) #f i (v @! i));
  fold (iarray_pts_to (from_array (raw_view #len) a) #f (fun i -> v @! i));
  ();
}

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
{
  iarray_begin_ a;
  (from_array (raw_view #len) a);
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
  let ggg = len;
  (* WOW AGAIN!!!! *)
  forevery_ext
    (fun (i : natlt len) ->
      iarray_pts_to_cell a #f i (v i))
    (fun (i : natlt len) ->
      iarray_pts_to_cell a #f i (Seq.init_ghost len v @! i));
  forevery_tostar #(natlt len) _;
  with sz p.
    rewrite bigstar 0 sz  p
         as bigstar 0 len p;
  B.gpu_array_unslice_1 (core a);
  ()
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
    a' : gpu_array et len
  ensures
    a' |-> Frac f (Seq.init_ghost len v)
{
  iarray_end_ a;
  (core a);
}

ghost
fn iarray_end2_
  (#et : Type0)
  (#vw : aiview { is_full_view vw })
  (a : iarray et vw)
  (#f : perm)
  (#v : vw.sch.ait -> GTot et)
  requires
    a |-> Frac f v
  ensures
    core a |-> Frac f (Seq.init_ghost (len vw) (fun (i : natlt (len vw)) -> v (it_of_nat vw i)))
{
  unfold iarray_pts_to a #f v;
  (* prove later... this should be fine. *)
  admit();
}

inline_for_extraction noextract
fn iarray_end2
  (#et : Type0)
  (#vw : aiview { is_full_view vw })
  (a : iarray et vw)
  (#f : perm)
  (#v : vw.sch.ait -> GTot et)
  requires
    a |-> Frac f v
  returns
    a' : gpu_array et (len vw)
  ensures
    a' |-> Frac f (Seq.init_ghost (len vw) (fun (i : natlt (len vw)) -> v (it_of_nat vw i)))
{
  iarray_end2_ a;
  (core a)
}

ghost
fn iarray_reindex_
  (#et : Type0)
  (#vw : aiview)
  (#ait' : Type)
  {| Enumerable.enumerable ait' |}
  (bij : vw.sch.ait =~ ait')
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.sch.ait -> GTot et))
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
  forevery_rw_type ait' (reindex_view vw bij).sch.ait _;
  fold iarray_pts_to
    (from_array (reindex_view vw bij) (core a)) #f
    (fun i -> v (bij.gg i));
  ()
}

fn iarray_reindex
  (#et : Type0)
  (#vw : aiview)
  (#ait' : Type)
  {| Enumerable.enumerable ait' |}
  (bij : vw.sch.ait =~ ait')
  (a : iarray et vw)
  (#f : perm)
  (#v : (vw.sch.ait -> GTot et))
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
  (#v : either vw1.sch.ait vw2.sch.ait -> GTot et)
  requires
    a |-> Frac f v
  ensures
    (from_array vw1 (core a) |-> Frac f (v `oo` Inl)) **
    (from_array vw2 (core a) |-> Frac f (v `oo` Inr))
{
  unfold iarray_pts_to a #f v;
  forevery_split_either #vw1.sch.ait #vw2.sch.ait _;

  forevery_ext #vw1.sch.ait
    (fun i -> iarray_pts_to_cell a #f (Inl i) (v (Inl i)))
    (fun i -> iarray_pts_to_cell (from_array vw1 (core a)) #f i (v (Inl #vw1.sch.ait #vw2.sch.ait i)));
  fold iarray_pts_to (from_array vw1 (core a)) #f (fun i -> v (Inl i));

  forevery_ext #vw2.sch.ait
    (fun i -> iarray_pts_to_cell a #f (Inr i) (v (Inr i)))
    (fun i -> iarray_pts_to_cell (from_array vw2 (core a)) #f i (v (Inr #vw1.sch.ait #vw2.sch.ait i)));
  fold iarray_pts_to (from_array vw2 (core a)) #f (fun i -> v (Inr i));
}

fn iarray_split2
  (#et:Type0)
  (vw1 vw2 : aiview { len vw1 == len vw2 }) // needed?
  (#_ : squash (no_overlap vw1.step.imap.f vw2.step.imap.f))
  (a : iarray et (sum_aiview vw1 vw2 #())) /// argh!!!! affects typeclass resolution!!!!
  (#f : perm)
  (#v : either vw1.sch.ait vw2.sch.ait -> GTot et)
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
fn iarray_share_n
  (#et:Type0)
  (#vw : aiview)
  (#[T.exact (`0)] uid: int)
  (a : iarray et vw)
  (k : pos)
  (#f : perm)
  (#v : vw.sch.ait -> GTot et)
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
fn iarray_gather_n
  (#et:Type0)
  (#vw : aiview)
  (#[T.exact (`0)] uid: int)
  (a : iarray et vw)
  (k : pos)
  (#f : perm)
  (#v : vw.sch.ait -> GTot et)
  requires
    bigstar #uid 0 k (fun _ -> a |-> Frac (f /. k) v)
  ensures
    a |-> Frac f v
{
  (* Boring *)
  admit();
}

inline_for_extraction noextract
fn iarray_write_cell
  (#et:Type0)
  (#vw : aiview) {| cw : ciview vw |}
  (a : iarray et vw)
  (ci : cw.sch.cit)
  (v1 : et)
  (#v0 : erased et)
  preserves gpu
  requires
    Cell a (ci_to_ai vw ci) |-> v0
  ensures
    Cell a (ci_to_ai vw ci) |-> v1
{
  let ai : erased vw.sch.ait = ci |> cw.sch.bij.gg; (* abstract index *)
  let ni = ci |~> cw.step.cimap;                    (* numerical index *)
  rewrite each ci_to_ai vw ci as ai;

  cw.step.compat ai;
  assert pure ((cw.step.cimap.f (cw.sch.bij.ff ai)) == SZ.uint_to_t (vw.step.imap.f ai));
  assert pure (SZ.v (cw.step.cimap.f (cw.sch.bij.ff ai)) == vw.step.imap.f ai);

  unfold iarray_pts_to_cell a ai v0;
  rewrite each (reveal ai |~> vw.step.imap) as ni;
  B.gpu_array_write #_ #_ #ni #(ni+1) (core a) ni v1;
  with s'. assert (B.gpu_pts_to_slice (core a) ni (ni+1) s');
  Kuiper.Seq.Common.lem_one_elem s' v1;
  rewrite each SZ.v ni as (ci_to_ai vw ci |~> vw.step.imap);
  fold iarray_pts_to_cell a (ci_to_ai vw ci) v1;
  ();
}

inline_for_extraction noextract
fn iarray_write_cell'
  (#et:Type0)
  (#vw : aiview) {| cw : ciview vw |}
  (a : iarray et vw)
  (ai : erased vw.sch.ait)
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
  preserves
    gpu
  requires
    iarray_pts_to_cell a #f (ci_to_ai vw ci) v0
  returns
    v : et
  ensures
    iarray_pts_to_cell a #f (ci_to_ai vw ci) v **
    pure (v == v0)
{
  let ai : erased vw.sch.ait = ci |> cw.sch.bij.gg; (* abstract index *)
  let ni = ci |~> cw.step.cimap;                    (* numerical index *)
  rewrite each ci_to_ai vw ci as ai;

  cw.step.compat ai;
  assert pure ((cw.step.cimap.f (cw.sch.bij.ff ai)) == SZ.uint_to_t (vw.step.imap.f ai));
  (* ^ FIXME: this should be exactly what we get from the line above? *)
  assert pure (SZ.v (cw.step.cimap.f (cw.sch.bij.ff ai)) == vw.step.imap.f ai);

  unfold iarray_pts_to_cell a #f ai v0;
  rewrite each (reveal ai |~> vw.step.imap) as ni;
  let res = B.gpu_array_read #_ #_ #ni #(ni+1) (core a) ni;
  with s'. assert (B.gpu_pts_to_slice (core a) #f ni (ni+1) s');
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
  (ai : erased vw.sch.ait)
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
  (#v : (vw.sch.ait -> GTot et))
  preserves
    gpu **
    (a |-> Frac f v)
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
  (#v0 : (vw.sch.ait -> GTot et))
  preserves
    gpu
  requires
    (a |-> v0)
  ensures
    (a |-> oplus v0 (ci_to_ai vw ci) e)
{
  unfold iarray_pts_to a v0;
  forevery_extract_if (ci_to_ai vw ci) _;
  iarray_write_cell a ci e;

  forevery_intro_if (ci_to_ai vw ci) (fun i -> iarray_pts_to_cell a i e);
  forevery_zip #vw.sch.ait
    (fun i -> if Enumerable.to_nat i = Enumerable.to_nat (ci_to_ai vw ci)
              then iarray_pts_to_cell a i e
              else emp)
    _;
  ghost
  fn aux (i : vw.sch.ait)
    requires
      (if Enumerable.to_nat i = Enumerable.to_nat (ci_to_ai vw ci)
       then iarray_pts_to_cell a i e
       else emp)
    **
      (if Enumerable.to_nat i = Enumerable.to_nat (ci_to_ai vw ci)
       then emp
       else iarray_pts_to_cell a i (v0 i))
    ensures
       iarray_pts_to_cell a i (oplus v0 (ci_to_ai vw ci) e i)
  {
    let cond = Enumerable.to_nat i = Enumerable.to_nat (ci_to_ai vw ci);
    if cond {
      rewrite each (ci_to_ai vw ci) as i;
      assert (pure (e == oplus v0 i e i));
      ();
    } else {
      rewrite each (Enumerable.to_nat i = Enumerable.to_nat (ci_to_ai vw ci)) as false;
      ();
    };
  };
  forevery_map _ _ aux;
  fold iarray_pts_to a (oplus v0 (ci_to_ai vw ci) e);
  ()
}
