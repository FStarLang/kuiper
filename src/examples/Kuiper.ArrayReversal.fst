module Kuiper.ArrayReversal
#lang-pulse
open Pulse.Lib
open Pulse.Lib.Pervasives
open Kuiper
open Kuiper.Bijection
open FStar.SizeT
module SZ = Kuiper.SizeT

noextract
let index_flip (#a:Type) (s:seq a) (i:nat { i < len s }) = Seq.index s (Seq.length s - i - 1 <: nat)
noextract
let reverse_spec (#a:Type) (s:seq a) : GTot _ = Seq.init (len s) (fun i -> index_flip s i)

let partition (n: nat { n % 2 == 0 }) : (natlt n =~ (natlt (n / 2) & bool)) =
  {
    ff = (fun i -> (if i < n / 2 then i else n - i - 1), i < n / 2)
      <: (natlt n -> (natlt (n / 2) & bool));
    gg = (fun x -> if snd x then fst x else n - fst x - 1)
      <: ((natlt (n / 2) & bool) -> natlt n);
    ff_gg = ez;
    gg_ff = ez;
  }

let gpu_pts_to_cell
  (#a:Type0)
  (#sz:nat)
  ([@@@mkey] arr : gpu_array a sz)
  (#f : perm)
  ([@@@mkey] i:nat)
  (v:a)
: slprop
= exists* s. gpu_pts_to_slice arr #f i (i+1) s **
             pure (s `Seq.equal` Seq.create 1 v)

ghost
fn explode_cells
  (#a:Type0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#s : lseq a sz)
requires
  arr |-> Frac f s
ensures
  forall+ (i: natlt sz). gpu_pts_to_cell arr #f i (Seq.index s i)
{
  gpu_array_slice_1 arr;
  forevery_map #(natlt sz)
    (fun (i: nat { i < sz }) -> gpu_pts_to_slice arr #f i (i+1) seq![Seq.index s i])
    (fun i -> gpu_pts_to_cell arr #f i (Seq.index s i))
    fn i {
      fold (gpu_pts_to_cell arr #f i (Seq.index s i))
    };
}

ghost
fn implode_cells
  (#a:Type0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#s : lseq a sz)
requires
  forall+ (i:natlt sz). gpu_pts_to_cell arr #f i (Seq.index s i)
ensures
  arr |-> Frac f s
{
  forevery_map #(natlt sz)
    (fun i -> gpu_pts_to_cell arr #f i (Seq.index s i))
    (fun i -> gpu_pts_to_slice arr #f i (Prims.op_Addition i 1) seq![Seq.index s i])
    fn i {
      unfold (gpu_pts_to_cell arr #f i (Seq.index s i));
      with s'. assert (gpu_pts_to_slice arr #f i (Prims.op_Addition i 1) s');
      assert pure (reveal s' `Seq.equal` seq![Seq.index (reveal s) i]);
    };
  gpu_array_unslice_1 arr #f #s
}

ghost
fn partition_cells
  (#a:Type0)
  (#size:sz { size % 2sz == 0sz })
  (arr : gpu_array a size)
  (#f : perm)
  (#s:seq a { len s == SZ.v size })
requires
  forall+ (i: natlt size).
    gpu_pts_to_cell arr #f i (Seq.index s i)
ensures
  forall+ (i: natlt (v size / 2)).
    gpu_pts_to_cell arr #f i (Seq.index s i) **
    gpu_pts_to_cell arr #f (SZ.v size - i - 1) (index_flip s i)
{
  forevery_iso (partition (SZ.v size)) _;
  forevery_unflatten' (fun (y: natlt (v size / 2) & bool) ->
    gpu_pts_to_cell arr #f
      ((partition (v size)).gg y)
      (Seq.Base.index s ((partition (v size)).gg y)));
  forevery_map #(natlt (v size / 2))
    (fun (x: natlt (v size / 2)) -> forall+ (y: bool).
      gpu_pts_to_cell arr #f
        ((partition (v size)).gg (x, y))
        (Seq.Base.index s ((partition (v size)).gg (x, y))))
    (fun (i: natlt (v size / 2)) ->
      gpu_pts_to_cell arr #f i (Seq.index s i) **
      gpu_pts_to_cell arr #f (SZ.v size - i - 1) (index_flip s i))
    fn i {
      forevery_bool_elim _;
      rewrite
        gpu_pts_to_cell arr #f
          ((partition (v size)).gg (i, true))
          (Seq.Base.index s ((partition (v size)).gg (i, true)))
      as
        gpu_pts_to_cell arr #f i (Seq.index s i);
      rewrite
        gpu_pts_to_cell arr #f
          ((partition (v size)).gg (i, false))
          (Seq.Base.index s ((partition (v size)).gg (i, false)))
      as
        gpu_pts_to_cell arr #f (SZ.v size - i - 1) (index_flip s i);
    };
  // forevery_rw_type (natlt (v size / 2)) (natlt (SZ.v (size `div` 2sz))) _;
}

ghost
fn partition_cells_inv
 (#a:Type0)
  (#size:sz { size % 2sz == 0sz })
  (arr : gpu_array a size)
  (#f : perm)
  (#s:seq a { len s == SZ.v size })
requires
  forall+ (i: natlt (SZ.v size / 2)).
    gpu_pts_to_cell arr #f i (Seq.index s i) **
    gpu_pts_to_cell arr #f (SZ.v size - i - 1) (index_flip s i)
ensures
  forall+ (i: natlt (SZ.v size)).
    gpu_pts_to_cell arr #f i (Seq.index s i)
{
  forevery_map #(natlt (v size / 2))
    (fun (i: natlt (v size / 2)) ->
      gpu_pts_to_cell arr #f i (Seq.index s i) **
      gpu_pts_to_cell arr #f (SZ.v size - i - 1) (index_flip s i))
    (fun (x: natlt (v size / 2)) -> forall+ (y: bool).
      gpu_pts_to_cell arr #f
        ((partition (v size)).gg (x, y))
        (Seq.Base.index s ((partition (v size)).gg (x, y))))
    fn i {
      rewrite
        gpu_pts_to_cell arr #f i (Seq.index s i)
      as
        gpu_pts_to_cell arr #f
          ((partition (v size)).gg (i, true))
          (Seq.Base.index s ((partition (v size)).gg (i, true)));
      rewrite
        gpu_pts_to_cell arr #f (SZ.v size - i - 1) (index_flip s i)
      as
        gpu_pts_to_cell arr #f
          ((partition (v size)).gg (i, false))
          (Seq.Base.index s ((partition (v size)).gg (i, false)));
      forevery_bool_intro (fun y ->
        gpu_pts_to_cell arr #f
          ((partition (v size)).gg (i, y))
          (Seq.Base.index s ((partition (v size)).gg (i, y))));
    };
  forevery_flatten _;
  forevery_iso_back (partition (v size)) (fun i ->
    gpu_pts_to_cell arr #f i (Seq.index s i));
}

inline_for_extraction
fn read_cell
  (#ty:Type0)
  (#size:erased sz)
  (a:gpu_array ty (SZ.v size))
  (i:sz { i < v size })
  (#p:perm)
  (#u:erased ty)
requires
  gpu **
  gpu_pts_to_cell a #p (SZ.v i) u
returns v:ty
ensures
  gpu **
  gpu_pts_to_cell a #p (SZ.v i) u **
  pure (u == v)
{
  unfold gpu_pts_to_cell;
  let v = gpu_array_read a i;
  fold (gpu_pts_to_cell a #p (SZ.v i) u);
  v
}

inline_for_extraction
fn write_cell
  (#ty:Type0)
  (#size:erased sz)
  (a:gpu_array ty (SZ.v size))
  (i:sz { i < v size })
  (v:ty)
  (#u:erased ty)
requires
  gpu **
  gpu_pts_to_cell a #1.0R (SZ.v i) u
ensures
  gpu **
  gpu_pts_to_cell a #1.0R (SZ.v i) v
{
  unfold gpu_pts_to_cell;
  gpu_array_write a i v;
  fold (gpu_pts_to_cell a #1.0R (SZ.v i) v);
}

unfold
let kpre
  (#ty:Type0)
  (size:sz)
  (a:gpu_array ty size)
  (s:seq ty{ len s == SZ.v size })
  (bid : natlt (SZ.v size / 2))
  : slprop =
  gpu_pts_to_cell a #1.0R bid (Seq.index s bid) **
  gpu_pts_to_cell a #1.0R (SZ.v size - bid - 1) (index_flip s bid)

unfold
let kpost
  (#ty:Type0)
  (size:sz)
  (a:gpu_array ty size)
  (s:seq ty{ len s == SZ.v size })
  (bid : natlt (SZ.v size / 2))
  : slprop =
  gpu_pts_to_cell a #1.0R bid (Seq.index (reverse_spec s) bid) **
  gpu_pts_to_cell a #1.0R (SZ.v size - bid - 1) (index_flip (reverse_spec s) bid)

inline_for_extraction noextract
fn kf
  (#ty:Type0)
  (size:sz)
  (a:gpu_array ty size)
  (#s:erased (Seq.seq ty) { len s == SZ.v size })
  (bid : szlt (SZ.v (size `div` 2sz))) (* pretty awful.. *)
  ()
  norewrite
  requires
    gpu **
    kpre size a s bid **
    block_id (size /^ 2sz) bid
  ensures
    gpu **
    kpost size a s bid **
    block_id (size /^ 2sz) bid
{
  let idx = bid; rewrite each bid as idx;
  let idx' = (size -^ idx -^ 1sz);
  rewrite each (SZ.v size - SZ.v idx - 1) as idx';
  let uu = read_cell a idx;
  let vv = read_cell a idx';
  write_cell a idx vv;
  write_cell a idx' uu;
  rewrite each SZ.v idx' as (SZ.v size - SZ.v idx - 1);
  rewrite each idx as bid;
  ()
}

ghost
fn setup
  (#ty:Type0)
  (size:sz { size > 0sz /\ size % 2sz == 0sz /\ SZ.v size < reveal max_blocks })
  (a:gpu_array ty size)
  (#s: erased (FStar.Seq.seq ty) { len s == SZ.v size })
  ()
  norewrite
  requires
    a |-> s
  ensures
    (forall+ (bid : natlt (size /^ 2sz)). kpre size a s bid) **
    emp (* frame *)
{
  explode_cells a;
  partition_cells a;
  forevery_rw_type (natlt (size / 2)) (natlt (size /^ 2sz)) _;
}

ghost
fn teardown
  (#ty:Type0)
  (size:sz { size > 0sz /\ size % 2sz == 0sz /\ SZ.v size < reveal max_blocks })
  (a:gpu_array ty size)
  (#s: erased (FStar.Seq.seq ty) { len s == SZ.v size })
  ()
  norewrite
  requires
    (forall+ (bid : natlt (size /^ 2sz)). kpost size a s bid) **
    emp (* frame *)
  ensures
    a |-> reverse_spec s
{
  forevery_rw_type (natlt (size /^ 2sz)) (natlt (size / 2)) _;
  partition_cells_inv a;
  implode_cells a
}

(* TODO: if size is odd, just put the middle cell in the frame! *)
inline_for_extraction noextract
let kdesc
  (#ty:Type0)
  (size:sz { size > 0sz /\ size %^ 2sz == 0sz /\ SZ.v size < reveal max_blocks })
  (a:gpu_array ty size)
  (#s: erased (FStar.Seq.seq ty) { len s == SZ.v size })
  : kernel_desc_m_1
      (a |-> s)
      (a |-> reverse_spec s)
  = {
      nblk     = size /^ 2sz;
      f        = kf size a #s;
      setup    = setup size a;
      teardown = teardown size a;
      kpre     = kpre size a s;
      kpost    = kpost size a s;
      frame    = emp;
  }

inline_for_extraction noextract
fn reverse
    (#ty:Type0)
    (size:sz { size > 0sz /\ size % 2sz == 0sz /\ SZ.v size < max_blocks })
    (a:gpu_array ty size)
    (#s: erased (seq ty) { len s == SZ.v size })
  preserves
    cpu
  requires
    a |-> s
  ensures
    a |-> reverse_spec s
{
  launch_sync (kdesc size a #s);
}

let reverse_u64 = reverse #u64
