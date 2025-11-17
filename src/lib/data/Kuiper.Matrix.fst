module Kuiper.Matrix
#lang-pulse

open Kuiper
open Kuiper.Bijection
open Kuiper.EMatrix
module A = Kuiper.VArray
module T = FStar.Tactics.V2

let gpu_matrix (et:Type0) (#rows #cols : nat) (l : mlayout rows cols) : Type0 =
  A.varray (aview_from_mlayout et #rows #cols l)

let is_global_matrix
  (#et:Type0) (#rows #cols : nat)
  (#l : mlayout rows cols)
  (arr: gpu_matrix et l)
: prop
= A.is_global_varray (arr)

let from_array l p = A.from_array (aview_from_mlayout _ l) p
let core g = A.core g

let lem_core_from_array
  (#et : Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et l)
  : Lemma (ensures from_array l (core g) == g)
          [SMTPat (core g)]
  = ()

let lem_from_array_core
  (#et : Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (p : gpu_array et (mlayout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let gpu_matrix_pts_to
  (#et:Type) (#rows #cols : nat)
  (#l : mlayout rows cols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  (em : ematrix et rows cols)
  : slprop
  = A.varray_pts_to gm #f em

instance is_send_across_global_matrix
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (x: gpu_matrix et l { is_global_matrix x })
  (#f : perm)
  (em : ematrix et rows cols)
: is_send_across gpu_of (gpu_matrix_pts_to x #f em)
= solve

ghost
fn gpu_matrix_pts_to_ref
  (#et:Type)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et l)
  (#f : perm)
  (#em : ematrix et rows cols)
  preserves
    gpu_matrix_pts_to g #f em
  ensures
    pure (SZ.fits (mlayout_size l))
{
  unfold gpu_matrix_pts_to g #f em;
  A.varray_pts_to_ref g;
  fold gpu_matrix_pts_to g #f em;
}

ghost
fn gpu_matrix_pts_to_ref_located
  (#et:Type)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et l)
  (#loc:_)
  (#f : perm)
  (#em : ematrix et rows cols)
  preserves
    on loc (gpu_matrix_pts_to g #f em)
  ensures
    pure (SZ.fits (mlayout_size l))
{
  ghost_impersonate loc
    (on loc (gpu_matrix_pts_to g #f em))
    (on loc (gpu_matrix_pts_to g #f em) ** pure (SZ.fits (mlayout_size l)))
    fn () {
      on_elim _;
      gpu_matrix_pts_to_ref g;
      on_intro (gpu_matrix_pts_to g #f em);
    }
}


// Sporadically fails
#push-options "--retry 3 --z3rlimit 20"
ghost
fn gpu_matrix_concr
  (#et:Type)
  (#rows #cols : nat)
  (#l : mlayout rows cols { is_full_layout l })
  (g : gpu_matrix et l)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    g |-> Frac f em
  ensures
    core g |-> Frac f (to_seq l em)
{
  unfold gpu_matrix_pts_to g #f em;
  A.varray_concr g;
  assert (pure (Seq.equal (to_seq l em) (A.to_seq (aview_from_mlayout et #rows #cols l) em)));
  rewrite A.core g |-> Frac f (A.to_seq (aview_from_mlayout et l) em)
       as core g |-> Frac f (to_seq l em);
  ()
}
#pop-options

// Sporadically fails
#push-options "--retry 3 --z3rlimit 20"
ghost
fn gpu_matrix_abs
  (#et:Type)
  (#rows #cols : nat)
  (l : mlayout rows cols { is_full_layout l })
  (p : gpu_array et (mlayout_size l))
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    p |-> Frac f (to_seq l em)
  ensures
    from_array l p |-> Frac f em
{
  assert (pure (Seq.equal (to_seq l em) (A.to_seq (aview_from_mlayout et l) em)));
  rewrite
    p |-> Frac f (to_seq l em)
  as
    p |-> Frac f (A.to_seq (aview_from_mlayout et l) em);
  // FIXME: does not work???
  //rewrite each to_seq l em as A.to_seq (aview_from_mlayout et l) em;
  A.varray_abs (aview_from_mlayout et l) p;
  fold gpu_matrix_pts_to (from_array l p) #f em;
}
#pop-options

ghost
fn gpu_matrix_abs'
  (#et:Type)
  (#rows #cols : nat)
  (l : mlayout rows cols { is_full_layout l })
  (p : gpu_array et (mlayout_size l))
  (#f : perm)
  (#s : lseq et (mlayout_size l))
  requires
    p |-> Frac f s
  ensures
    from_array l p |-> Frac f (from_seq l s)
{
  rewrite each s as to_seq l (from_seq l s);
  gpu_matrix_abs l p;
}

(* This version does not require a full_layout. *)
ghost
fn gpu_matrix_iconcr
  (#et:Type)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et l)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    g |-> Frac f em
  ensures
    pure (SZ.fits (mlayout_size l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      gpu_pts_to_cell (core g) #f (cell_of_pos l r c) (macc em r c))
{
  unfold gpu_matrix_pts_to g #f em;
  A.varray_pts_to_ref g;
  A.varray_iconcr g;

  forevery_rw_type _ (natlt rows & natlt cols) _;
  forevery_unflatten' _;
  forevery_ext_2 _
    (fun r c -> gpu_pts_to_cell (core g) #f (cell_of_pos l r c) (macc em r c));
}

ghost
fn gpu_matrix_iabs
  (#et:Type)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et l)
  (#em : ematrix et rows cols)
  (#f : perm)
  requires
    pure (SZ.fits (mlayout_size l)) **
    (forall+ (r : natlt rows) (c : natlt cols).
      gpu_pts_to_cell (core g) #f (cell_of_pos l r c) (macc em r c))
  ensures
    g |-> Frac f em
{
  forevery_flatten _;
  forevery_rw_type _ ((aview_from_mlayout et l).iview.ait) _;
  forevery_ext _
    (fun i -> gpu_pts_to_cell (A.core g) #f ((aview_from_mlayout et l).iview.step.imap.f i)
      ((aview_from_mlayout et l).ctn.acc em i));

  A.varray_iabs g;
  fold gpu_matrix_pts_to g #f em;
}

inline_for_extraction noextract
fn gpu_matrix_alloc0
  (#et:Type) {| sized et |}
  (rows cols : szp)
  (l : mlayout rows cols { is_full_layout l })
  preserves
    cpu
  requires
    pure (SZ.fits (rows * cols))
  returns
    gm : gpu_matrix et l
  ensures
    exists* em. on gpu_loc (gm |-> em)
  ensures   pure (is_global_matrix gm)
{
  open FStar.SizeT;
  let gm = A.varray_alloc0 (rows *^ cols) (aview_from_mlayout et l);
  gm;
}

#set-options "--print_implicits"

inline_for_extraction noextract
fn gpu_matrix_free
  (#et:Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols { is_full_layout l })
  (gm : gpu_matrix et l)
  (#em : _)
  preserves
    cpu
  requires
    on gpu_loc (gm |-> em)
  ensures emp
{
  A.varray_free gm;
}

ghost
fn gpu_matrix_share_n
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    forall+ (_:natlt k). gpu_matrix_pts_to gm #(f /. k) em
{
  unfold gpu_matrix_pts_to gm #f em;
  A.varray_share_n gm k;
  forevery_map
    (fun (i:natlt k) -> A.varray_pts_to gm #(f /. k) em)
    (fun (i:natlt k) -> gpu_matrix_pts_to gm #(f /. k) em)
    fn i { fold gpu_matrix_pts_to gm #(f /. k) em };
}

ghost
fn gpu_matrix_gather_n
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    forall+ (_:natlt k). gpu_matrix_pts_to gm #(f /. k) em
  ensures
    gpu_matrix_pts_to gm #f em
{
  forevery_map
    (fun (i:natlt k) -> gpu_matrix_pts_to gm #(f /. k) em)
    (fun (i:natlt k) -> A.varray_pts_to gm #(f /. k) em)
    fn i { unfold gpu_matrix_pts_to gm #(f /. k) em };
  A.varray_gather_n gm k;
  fold gpu_matrix_pts_to gm #f em;
}

ghost
fn gpu_matrix_pts_to_eq
  (#et : Type u#0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (m : gpu_matrix et l)
  (#f1 f2 : perm)
  (#em1 #em2 : ematrix et rows cols)
  requires
    gpu_matrix_pts_to m #f1 em1 **
    gpu_matrix_pts_to m #f2 em2
  ensures
    gpu_matrix_pts_to m #f1 em2 **
    gpu_matrix_pts_to m #f2 em2
{
  unfold gpu_matrix_pts_to m #f1 em1;
  unfold gpu_matrix_pts_to m #f2 em2;
  A.varray_pts_to_eq m f2;
  fold gpu_matrix_pts_to m #f1 em2;
  fold gpu_matrix_pts_to m #f2 em2;
}

ghost
fn gpu_matrix_gather_n_underspec
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (k : pos)
  (#f : perm)
  requires
    forall+ (_ : natlt k).
      exists* (em: ematrix et rows cols). gpu_matrix_pts_to gm #(f /. k) em
  ensures
    exists* (em : ematrix et rows cols). gpu_matrix_pts_to gm #f em
{
  forevery_natlt_pop k _;
  with em. assert gpu_matrix_pts_to gm #(f /. k) em;
  ghost
  fn aux (_ : natlt (k-1))
    norewrite
    requires
      gpu_matrix_pts_to gm #(f /. k) em ** (exists* v. gpu_matrix_pts_to gm #(f /. k) v)
    ensures
      gpu_matrix_pts_to gm #(f /. k) em ** gpu_matrix_pts_to gm #(f /. k) em
  {
    gpu_matrix_pts_to_eq gm (f /. k) #_ #em;
  };
  forevery_map_extra #(natlt (k-1)) (gpu_matrix_pts_to gm #(f /. k) em)
    (fun (_ : natlt (k-1)) -> exists* v. gpu_matrix_pts_to gm #(f /. k) v)
    (fun (_ : natlt (k-1)) -> gpu_matrix_pts_to gm #(f /. k) em)
    aux;
  forevery_natlt_push k _;
  gpu_matrix_gather_n gm k;
}

ghost
fn gpu_matrix_share_2
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (#em : ematrix et rows cols)
  requires
    gm |-> em
  ensures
    (gm |-> Frac 0.5R em) ** (gm |-> Frac 0.5R em)
{
  unfold gpu_matrix_pts_to gm em;
  A.varray_share_n gm 2;
  forevery_natlt_pop 2 _;
  forevery_natlt_pop 1 _;
  forevery_elim_empty _;
  fold gpu_matrix_pts_to gm #0.5R em;
  fold gpu_matrix_pts_to gm #0.5R em;
}

ghost
fn gpu_matrix_gather_2
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (#em : ematrix et rows cols)
  requires
    (gm |-> Frac 0.5R em) ** (gm |-> Frac 0.5R em)
  ensures
    gm |-> em
{
  unfold gpu_matrix_pts_to gm #0.5R em;
  unfold gpu_matrix_pts_to gm #0.5R em;
  forevery_intro_empty #(natlt 0) (fun _ -> A.varray_pts_to gm #(1.0R /. 2) em);
  forevery_natlt_push 1 _;
  forevery_natlt_push 2 _;
  A.varray_gather_n gm 2;
  fold gpu_matrix_pts_to gm em;
}

inline_for_extraction noextract
fn gpu_matrix_read
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| cl : clayout l |}
  (gm : gpu_matrix et l)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu **
    gpu_matrix_pts_to gm #f em
  returns v : et
  ensures
    gpu **
    gpu_matrix_pts_to gm #f em **
    pure (v == macc em i j)
{
  unfold gpu_matrix_pts_to gm #f em;
  let r = A.varray_read gm (i, j);
  fold gpu_matrix_pts_to gm #f em;
  r
}

inline_for_extraction noextract
fn gpu_matrix_write
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l |}
  (gm : gpu_matrix et l)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (v : et)
  (#em : ematrix et rows cols)
  requires
    gpu **
    gpu_matrix_pts_to gm em
  ensures
    gpu **
    gpu_matrix_pts_to gm (mupd em i j v)
{
  unfold gpu_matrix_pts_to gm em;
  A.varray_write gm (i,j) v;
  assert (pure (
    mupd em i j v
    `Kuiper.EMatrix.equal`
    (aview_from_mlayout et #rows #cols l).ctn.upd em (A.ci_to_ai _ (i,j)) v));
  fold gpu_matrix_pts_to gm (mupd em i j v);
}

let gpu_matrix_pts_to_cell
  (#et:Type) (#rows #cols : nat)
  (#l : mlayout rows cols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : natlt rows)
  ([@@@mkey]j : natlt cols)
  (v : et)
  : slprop
  = A.varray_pts_to_cell gm #f (i,j) v

let gpu_matrix_pts_to_cell_eq
  (#et:Type) (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (i : natlt rows)
  (j : natlt cols)
  (f : perm)
  (v : et)
  : Lemma (gpu_matrix_pts_to_cell gm #f i j v
           ==
           gpu_pts_to_cell (core gm) #f (cell_of_pos l i j) v)
  = A.varray_pts_to_cell_eq gm (i,j) f v

inline_for_extraction noextract
fn gpu_matrix_read_cell
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| c : clayout l |}
  (gm : gpu_matrix et l)
  (i : szlt rows)
  (j : szlt cols)
  (#f : perm)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm #f i j v0
  returns v : et
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm #f i j v **
    pure (v == v0)
{
  unfold gpu_matrix_pts_to_cell gm #f i j v0;
  (* very awkward *)
  rewrite
    each Mktuple2 #(natlt rows) #(natlt cols) (SZ.v i) (SZ.v j)
      as A.ci_to_ai (aview_from_mlayout et l) (i, j);
  let v = A.varray_read_cell gm (i,j);
  with ai. assert (A.varray_pts_to_cell gm #f ai v0);
  rewrite each ai as Mktuple2 #(natlt rows) #(natlt cols) i j;
  fold gpu_matrix_pts_to_cell gm #f i j v0;
  v;
}

inline_for_extraction noextract
fn gpu_matrix_write_cell
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| c : clayout l |}
  (gm : gpu_matrix et l)
  (i : szlt rows)
  (j : szlt cols)
  (v1 : et)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm i j v0
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm i j v1
{
  unfold gpu_matrix_pts_to_cell gm i j v0;
  (* very awkward *)
  rewrite
    each Mktuple2 #(natlt rows) #(natlt cols) (SZ.v i) (SZ.v j)
      as A.ci_to_ai (aview_from_mlayout et l) (i, j);
  A.varray_write_cell gm (i,j) v1;
  with ai. assert (A.varray_pts_to_cell gm #1.0R ai v1);
  rewrite each ai as Mktuple2 #(natlt rows) #(natlt cols) i j;
  fold gpu_matrix_pts_to_cell gm i j v1;
}

ghost
fn gpu_matrix_explode
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    forall+ r c.
      gpu_matrix_pts_to_cell gm #f r c (macc em r c)
{
  unfold gpu_matrix_pts_to gm #f em;
  A.varray_explode gm;
  forevery_rw_type (aview_from_mlayout et l).iview.ait (natlt rows & natlt cols) _;
  ghost
  fn aux (rc : natlt rows & natlt cols)
    requires A.varray_pts_to_cell gm #f rc ((aview_from_mlayout et l).ctn.acc em rc)
    ensures  gpu_matrix_pts_to_cell gm #f rc._1 rc._2 (macc em rc._1 rc._2)
  {
    rewrite each rc as (rc._1, rc._2);
    fold gpu_matrix_pts_to_cell gm #f rc._1 rc._2 (macc em rc._1 rc._2);
  };
  forevery_map #(natlt rows & natlt cols)
    (fun rc -> A.varray_pts_to_cell gm #f rc ((aview_from_mlayout et l).ctn.acc em rc))
    (fun rc -> gpu_matrix_pts_to_cell gm #f rc._1 rc._2 (macc em rc._1 rc._2))
    aux;
  forevery_unflatten #(natlt rows) #(natlt cols) (fun r c ->
    gpu_matrix_pts_to_cell gm #f r c (macc em r c));
  ()
}

ghost
fn gpu_matrix_implode
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    pure (SZ.fits (mlayout_size l))
  requires
    forall+ r c.
      gpu_matrix_pts_to_cell gm #f r c (macc em r c)
  ensures
    gpu_matrix_pts_to gm #f em
{
  forevery_flatten #(natlt rows) #(natlt cols)
    (fun r c -> gpu_matrix_pts_to_cell gm #f r c (macc em r c));
  forevery_ext #(natlt rows & natlt cols)
    (fun i -> gpu_matrix_pts_to_cell gm #f i._1 i._2 (macc em i._1 i._2))
    (fun i -> A.varray_pts_to_cell gm #f i ((aview_from_mlayout et l).ctn.acc em i));
  A.varray_implode gm;
  fold gpu_matrix_pts_to gm #f em;
}

inline_for_extraction noextract
fn gpu_matrix_from_array
  (#et:Type0) {| sized et |}
  (#rows #cols : SZ.t)
  (#l : mlayout rows cols { is_full_layout l })
  (gm : gpu_matrix et l)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == rows * cols})
  (#em : ematrix et rows cols)
  preserves
    a |-> s **
    cpu
  requires
    on gpu_loc (gm |-> em)
  ensures
    pure (SZ.fits (rows * cols) /\ Pulse.Lib.Vec.length a == rows * cols) **
    on gpu_loc (gm |-> from_seq l s)
{
  Pulse.Lib.Vec.pts_to_len a;
  A.varray_from_array (rows *^ cols) gm a;
  from_seq_rel l s;
  with p. assert (on gpu_loc p);
  map_loc gpu_loc #p #(gpu_matrix_pts_to gm #1.0R (from_seq l s)) fn () {
    fold gpu_matrix_pts_to gm #1.0R (from_seq l s)
  };
}

inline_for_extraction noextract
fn gpu_matrix_to_array
  (#et:Type0) {| sized et |}
  (#rows #cols : SZ.t)
  (#l : mlayout rows cols { is_full_layout l })
  (a : vec et)
  (gm : gpu_matrix et l)
  (#s : erased (seq et){Seq.length s == rows * cols})
  (#em : ematrix et rows cols)
  preserves
    on gpu_loc (gm |-> em) **
    cpu
  requires
    a |-> s
  ensures
    pure (SZ.fits (rows * cols) /\ Pulse.Lib.Vec.length a == rows * cols) **
    (a |-> to_seq l em)
{
  Pulse.Lib.Vec.pts_to_len a;
  A.varray_to_array (rows *^ cols) a gm;
  to_seq_rel l em;
  ()
}
