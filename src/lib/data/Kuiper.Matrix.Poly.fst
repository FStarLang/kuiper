module Kuiper.Matrix.Poly
#lang-pulse

open Kuiper
open Kuiper.Bijection
module T = FStar.Tactics.V2

let to_from_inv (#et:Type) (#rows #cols : nat) (l : mlayout rows cols)
  (s : lseq et (rows * cols))
  : Lemma (to_seq l (from_seq l s) == s)
          [SMTPat (to_seq l (from_seq l s))]
  = assert (Seq.equal (to_seq l (from_seq l s)) s);
    ()

let seq_acc (#et:Type) (#rows #cols : nat)
  (l : mlayout rows cols)
  (m : ematrix et rows cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma (macc m i j == to_seq l m @! l.bij.ff (i,j))
          [SMTPat (macc m i j); SMTPat (to_seq l m)]
  = ()

let seq_upd (#et:Type) (#rows #cols : nat)
  (l : mlayout rows cols)
  (m : ematrix et rows cols)
  (i : natlt rows) (j : natlt cols)
  (v : et)
  : Lemma (to_seq l (mupd m i j v) == Seq.upd (to_seq l m) (l.bij.ff (i,j)) v)
          [SMTPat (mupd m i j v); SMTPat (to_seq l m)]
  = assert (Seq.equal (to_seq l (mupd m i j v)) (Seq.upd (to_seq l m) (l.bij.ff (i,j)) v));
    ()

let gpu_matrix (et:Type0) (rows cols : nat) (l : mlayout rows cols) : Type0 =
  gpu_array et (rows * cols)

let core g = g
let core_match g1 g2 = ()

let gpu_matrix_pts_to
  (#et:Type) (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  ([@@@mkey] gm : gpu_matrix et rows cols l)
  (#[T.exact (`1.0R)] f : perm)
  (em : ematrix et rows cols)
  : slprop
  = gpu_pts_to_array gm #f (to_seq l em)

inline_for_extraction noextract
fn gpu_matrix_concr
  (#et:Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (g : gpu_matrix et rows cols l)
  (#em : ematrix et rows cols)
  requires
    g |-> em
  ensures
    core g |-> to_seq l em
{
  unfold gpu_matrix_pts_to g #1.0R em;
}

inline_for_extraction noextract
fn gpu_matrix_abs
  (#et:Type)
  (#rows0 #cols0 : erased nat) (#l0 : mlayout rows0 cols0)
  (g : gpu_matrix et rows0 cols0 l0)
  (rows cols : erased nat) (l : mlayout rows cols)
  (#em : ematrix et rows cols)
  requires
    core g |-> to_seq l em
  returns
    g' : gpu_matrix et rows cols l
  ensures
    pure (rows * cols == rows0 * cols0 /\ core g == core g') **
    (g' |-> em)
{
  gpu_pts_to_ref (core g);
  let g' : gpu_array et (rows0 * cols0) = core g;
  rewrite each core g as g';
  fold gpu_matrix_pts_to #_ #_ #_ #l g' #1.0R em;
  g'
}

inline_for_extraction noextract
fn gpu_matrix_alloc
  (#et:Type) {| sized et |}
  (rows cols : szp)
  (l : mlayout rows cols)
  preserves
    cpu
  requires
    pure (SZ.fits (rows * cols))
  returns
    gm : gpu_matrix et rows cols l
  ensures
    exists* em. gm |-> em
{
  open FStar.SizeT;
  let gm = gpu_array_alloc #et (rows *^ cols);
  with s. assert (gpu_pts_to_array gm #1.0R s);
  let em = from_seq l s;
  fold (gpu_matrix_pts_to #_ #_ #_ #l gm em);
  gm;
}

inline_for_extraction noextract
fn gpu_matrix_free
  (#et:Type)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et rows cols l)
  (#em : _)
  preserves
    cpu
  requires
    gm |-> em
  ensures emp
{
  unfold gpu_matrix_pts_to gm em;
  gpu_array_free gm;
}

inline_for_extraction noextract
fn gpu_matrix_from_array
  (#et:Type) {| sized et |}
  (#rows #cols : szp)
  (#l : mlayout rows cols)
  (a : vec et)
  (gA : gpu_matrix et rows cols l)
  (#s : erased (seq et){ len s == rows * cols })
  preserves
    (a |-> s) **
    cpu
  requires
    (gA |-> 'm0) **
    pure (SZ.fits (rows * cols))
  ensures
    gA |-> from_seq l s
{
  unfold gpu_matrix_pts_to gA 'm0;
  Kuiper.Array.gpu_memcpy_host_to_device gA a (SZ.mul rows cols);
  fold gpu_matrix_pts_to gA (from_seq l s);
  ();
}

inline_for_extraction noextract
fn gpu_matrix_to_array
  (#et:Type) {| sized et |}
  (#rows #cols : szp)
  (#l : mlayout rows cols)
  (a : vec et)
  (gA : gpu_matrix et rows cols l)
  (#m : ematrix et rows cols)
  preserves
    (gA |-> m) **
    cpu
  requires
    (a |-> 's0) **
    pure (SZ.fits (rows * cols) /\ Pulse.Lib.Vec.length a == rows * cols)
  ensures
    a |-> to_seq l m
{
  Pulse.Lib.Vec.pts_to_len a;
  unfold gpu_matrix_pts_to gA m;
  Kuiper.Array.gpu_memcpy_device_to_host a gA (SZ.mul rows cols);
  fold gpu_matrix_pts_to gA m;
  ();
}

ghost
fn gpu_matrix_share_n
  (#et:Type0)
  (#uid: int)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et rows cols l)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    bigstar #uid 0 k (fun _ -> gpu_matrix_pts_to gm #(f /. k) em)
{
  admit(); // just tedious
}

ghost
fn gpu_matrix_gather_n
  (#et:Type0)
  (#uid: int)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et rows cols l)
  (k : pos)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    bigstar #uid 0 k (fun _ -> gpu_matrix_pts_to gm #(f /. k) em)
  ensures
    gpu_matrix_pts_to gm #f em
{
  admit(); // just tedious
}

(* NOTE: we cannot just call the projector, since we
need the concrete nats to do so (?!) and that would
incur a ghost effect. *)
inline_for_extraction noextract
let lcto (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| c : clayout l |}
  (i : SZ.t{SZ.v i < rows})
  (j : SZ.t{SZ.v j < cols})
  : r:SZ.t{r == c.c_to i j}
  = match c with
    | { c_to } -> c_to i j

inline_for_extraction noextract
fn gpu_matrix_read
  (#et:Type0)
  (#rows : erased nat)
  (#cols : erased nat)
  (#l : mlayout rows cols) {| c : clayout l |}
  (gm : gpu_matrix et rows cols l)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (#f:perm)
  (#em : ematrix et rows cols)
  requires
    gpu **
    gpu_matrix_pts_to gm #f em
  returns
    v : et
  ensures
    gpu **
    gpu_matrix_pts_to gm #f em **
    pure (v == macc em i j)
{
  open FStar.SizeT;
  unfold gpu_matrix_pts_to gm #f em;
  gpu_pts_to_ref gm;
  let idx : sz = lcto #_ #_ #_ #c i j;
  let v = gpu_array_read #et #(rows * cols) #0 #(rows * cols) gm idx;
  fold gpu_matrix_pts_to gm #f em;
  v;
}

inline_for_extraction noextract
fn gpu_matrix_write
  (#et:Type0)
  (#rows : erased nat)
  (#cols : erased nat)
  (#l : mlayout rows cols) {| c : clayout l |}
  (gm : gpu_matrix et rows cols l)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (vv : et)
  (#em : ematrix et rows cols)
  requires
    gpu **
    gpu_matrix_pts_to gm em
  ensures
    gpu **
    gpu_matrix_pts_to gm (mupd em i j vv)
{
  open FStar.SizeT;
  unfold gpu_matrix_pts_to gm em;
  gpu_pts_to_ref gm;
  let idx = lcto #_ #_ #_ #c i j;
  gpu_array_write #et #(rows * cols) #0 #(rows * cols) gm idx vv;
  fold gpu_matrix_pts_to gm (mupd em i j vv);
  ();
}

let gpu_matrix_pts_to_cell
  (#et:Type) (#rows #cols : nat)
  (#l : mlayout rows cols)
  ([@@@mkey] gm : gpu_matrix et rows cols l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : natlt rows)
  ([@@@mkey]j : natlt cols)
  (v : et)
  : slprop
  = gpu_pts_to_slice gm #f (l.bij.ff (i,j)) (l.bij.ff (i,j) + 1) seq![v]

inline_for_extraction noextract
fn gpu_matrix_read_cell
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| c : clayout l |}
  (gm : gpu_matrix et rows cols l)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
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
  open FStar.SizeT;
  unfold gpu_matrix_pts_to_cell gm #f i j v0;
  gpu_pts_to_slice_ref #et #(rows * cols) #f gm _ _ #(seq![reveal v0]);
  let idx = lcto #_ #_ #_ #c i j;
  let v = gpu_array_read #et #(rows * cols) #idx #(idx+1) gm idx;
  fold gpu_matrix_pts_to_cell gm #f i j v0;
  v;
}

inline_for_extraction noextract
fn gpu_matrix_write_cell
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| c : clayout l |}
  (gm : gpu_matrix et rows cols l)
  (i : sz{SZ.v i < rows})
  (j : sz{SZ.v j < cols})
  (v1 : et)
  (#v0 : erased et)
  requires
    gpu **
    gpu_matrix_pts_to_cell gm i j v0
  ensures
    gpu **
    gpu_matrix_pts_to_cell gm i j v1
{
  open FStar.SizeT;
  unfold gpu_matrix_pts_to_cell gm i j v0;
  gpu_pts_to_slice_ref #et #(rows * cols) gm _ _ #(seq![reveal v0]);
  let idx = lcto #_ #_ #_ #c i j;
  assert (gpu_pts_to_slice gm idx (idx+1) seq![reveal v0]);
  gpu_array_write #et #(rows * cols) #idx #(idx+1) gm idx v1;
  with s'. assert (gpu_pts_to_slice gm idx (idx+1) s');
  Kuiper.Seq.Common.lem_one_elem s' v1;
  assert (gpu_pts_to_slice gm idx (idx+1) seq![v1]);
  fold gpu_matrix_pts_to_cell gm i j v1;
}

ghost
fn gpu_matrix_explode
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et rows cols l)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    gpu_matrix_pts_to gm #f em
  ensures
    forall+ r c.
      gpu_matrix_pts_to_cell gm #f r c (macc em r c)
{
  unfold gpu_matrix_pts_to gm #f em;
  gpu_array_slice_1 gm;
  assert bigstar 0 (rows * cols) (fun i -> gpu_pts_to_slice gm #f i (i+1) seq![Seq.index (to_seq l em) i]);
  rewrite
    bigstar 0 (rows * cols) (fun i -> gpu_pts_to_slice gm #f i (i+1) seq![Seq.index (to_seq l em) i])
  as
    bigstar 0 (Kuiper.Enumerable.cardinal (natlt (rows * cols)))
      (fun i -> gpu_pts_to_slice gm #f i (i+1) seq![Seq.index (to_seq l em) i]);
  forevery_fromstar #(natlt (rows * cols))
    (fun i -> gpu_pts_to_slice gm #f i (i+1) seq![Seq.index (to_seq l em) i]);
  forevery_iso (bij_sym l.bij) _;
  forevery_unflatten' #(natlt rows) #_ #(natlt cols) _;
  forevery_ext_2
    #(natlt rows) #_
    #(natlt cols) #_
    (fun r c -> gpu_pts_to_slice gm #f (l.bij.ff (r,c)) (l.bij.ff (r,c) + 1) seq![Seq.index (to_seq l em) (l.bij.ff (r,c))])
    (fun r c -> gpu_pts_to_slice gm #f (l.bij.ff (r,c)) (l.bij.ff (r,c) + 1) seq![macc em r c]);
}

ghost
fn gpu_matrix_implode
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et rows cols l)
  (#f : perm)
  (#em : ematrix et rows cols)
  requires
    forall+ r c.
      gpu_matrix_pts_to_cell gm #f r c (macc em r c)
  ensures
    gpu_matrix_pts_to gm #f em
{
  forevery_ext_2
    #(natlt rows) #_
    #(natlt cols) #_
    (fun r c -> gpu_pts_to_slice gm #f (l.bij.ff (r,c)) (l.bij.ff (r,c) + 1) seq![macc em r c])
    (fun r c -> gpu_pts_to_slice gm #f (l.bij.ff (r,c)) (l.bij.ff (r,c) + 1) seq![Seq.index (to_seq l em) (l.bij.ff (r,c))]);
  forevery_flatten' #(natlt rows) #_ #(natlt cols)
    (fun (x,y) -> gpu_pts_to_slice gm #f (l.bij.ff (x,y)) (l.bij.ff (x,y) + 1) seq![Seq.index (to_seq l em) (l.bij.ff (x,y))]);
  forevery_iso l.bij _;
  forevery_ext
    #(natlt (rows * cols))
    (fun i -> let x,y = l.bij.gg i in
              gpu_pts_to_slice gm #f (l.bij.ff (x,y)) (l.bij.ff (x,y) + 1) seq![Seq.index (to_seq l em) (l.bij.ff (x,y))])
    (fun y -> gpu_pts_to_slice gm #f y (y + 1) seq![Seq.index (to_seq l em) y]);
  forevery_tostar #(natlt (rows * cols))
    (fun i -> gpu_pts_to_slice gm #f i (i+1) seq![Seq.index (to_seq l em) i]);
  rewrite
    bigstar 0 (Kuiper.Enumerable.cardinal (natlt (rows * cols)))
      (fun i -> gpu_pts_to_slice gm #f i (i+1) seq![Seq.index (to_seq l em) i])
  as
    bigstar 0 (rows * cols) (fun i -> gpu_pts_to_slice gm #f i (i+1) seq![Seq.index (to_seq l em) i]);
  assert bigstar 0 (rows * cols) (fun i -> gpu_pts_to_slice gm #f i (i+1) seq![Seq.index (to_seq l em) i]);
  gpu_array_unslice_1 gm;
  fold gpu_matrix_pts_to gm #f em;
}
