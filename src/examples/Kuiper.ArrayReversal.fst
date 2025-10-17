module Kuiper.ArrayReversal
#lang-pulse
open FStar.Tactics
open Pulse.Lib
open Pulse.Lib.Pervasives
open Kuiper
open Pulse.Lib.BoundedIntegers
open Pulse.Lib.PartitionRange
open FStar.SizeT
open FStar.FiniteSet.Base
module Set = FStar.FiniteSet.Base
module SZ = Kuiper.SizeT

noextract
let index_flip (#a:Type) (s:seq a) (i:nat { i < len s }) = Seq.index s (Seq.length s - i - 1 <: nat)
noextract
let reverse_spec (#a:Type) (s:seq a) : GTot _ = Seq.init (len s) (fun i -> index_flip s i)

noextract
let partition_range (n:nat { n % 2 == 0 })
: partitions 0 n (n / 2)
= fun i ->
    FStar.FiniteSet.Base.all_finite_set_facts_lemma();
    Set.union (Set.singleton i) (Set.singleton (n - i - 1))

noextract
let star_over_partition_range (n:nat { n % 2 == 0 }) (f:idx 0 n -> slprop) (i:nat { i < n / 2 })
: Lemma
  (ensures star_over_partition f (select (partition_range n) i) ==
            f i ** f (n - i - 1 <: nat))
= FStar.FiniteSet.Base.all_finite_set_facts_lemma();
  slprop_equivs()

noextract
let partition_range_disjoint (n:nat { n % 2 == 0 })
: Lemma (parts_disjoint (partition_range n))
= FStar.FiniteSet.Base.all_finite_set_facts_lemma()

#restart-solver
#push-options "--fuel 0 --ifuel 0"
let partition_covers_range_lemma (n:nat { n % 2 == 0 })
: Lemma (ensures parts_covers_range (partition_range n))
= FStar.FiniteSet.Base.all_finite_set_facts_lemma();
  introduce forall z. z `Set.mem` range 0 n ==> z `Set.mem` union_partitions (partition_range n)
  with introduce _ ==> _
  with _. (
    if z < n / 2
    then (
      assert (z `Set.mem` select (partition_range n) z)
    )
    else (
      assert (z `Set.mem` select (partition_range n) (n - z - 1 <: nat))
    )
  )
#pop-options

noextract
let disjoint_partitions_range (n:nat { n % 2 == 0 })
: disjoint_partitions 0 n (n / 2)
= partition_range_disjoint n;
  partition_covers_range_lemma n;
  partition_range n

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
  bigstar 0 sz (fun i -> gpu_pts_to_cell arr #f i (Seq.index s i))
{
  gpu_array_slice_1 arr;
  ghost
  fn slice_to_cell (i:nat { 0 <= i /\ i < sz })
  requires
    gpu_pts_to_slice arr #f i (i+1) seq![Seq.index s i]
  ensures
    gpu_pts_to_cell arr #f i (Seq.index s i)
  {
    fold (gpu_pts_to_cell arr #f i (Seq.index s i))
  };
  bigstar_map
    #_ #_ #0 #sz
    #(fun i -> gpu_pts_to_slice arr #f i (i+1) seq![Seq.index s i])
    #(fun i -> gpu_pts_to_cell arr #f i (Seq.index s i))
    slice_to_cell;
}


ghost
fn implode_cells
  (#a:Type0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#s : lseq a sz)
requires
  bigstar 0 sz (fun (i:idx 0 sz) -> gpu_pts_to_cell arr #f i (Seq.index s i))
ensures
  arr |-> Frac f s
{
  ghost
  fn cell_to_slice (i:nat { 0 <= i /\ i < sz })
  requires
    gpu_pts_to_cell arr #f i (Seq.index s i)
  ensures
    gpu_pts_to_slice arr #f i (Prims.op_Addition i 1) seq![Seq.index s i]
  {
    unfold (gpu_pts_to_cell arr #f i (Seq.index s i));
    with s'. assert (gpu_pts_to_slice arr #f i (i+1) s');
    assert pure (reveal s' `Seq.equal` seq![Seq.index (reveal s) i]);
  };
  bigstar_map
    #_ #_ #0 #sz
    #(fun (i:idx 0 sz) -> gpu_pts_to_cell arr #f i (Seq.index s i))
    #(fun i -> gpu_pts_to_slice arr #f i (Prims.op_Addition i 1) seq![Seq.index s i])
    cell_to_slice;
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
  bigstar 0 size (fun i ->
    gpu_pts_to_cell arr #f i (Seq.index s i))
ensures
  bigstar 0 (SZ.v (size `div` 2sz)) (fun i ->
    gpu_pts_to_cell arr #f i (Seq.index s i) **
    gpu_pts_to_cell arr #f (SZ.v size - i - 1) (index_flip s i))
{
  bigstar_partition size (size `div` 2sz)
    (fun i -> gpu_pts_to_cell arr #f i (Seq.index s i))
    (disjoint_partitions_range size);

  ghost
  fn star_over_partition_range
      (n:nat { n % 2 == 0 })
      (f:idx 0 n -> slprop)
      (i:nat { 0 <= i /\ i < n / 2 })
  requires
    star_over_partition f (select (disjoint_partitions_range n) i)
  ensures
    f i ** f (n - i - 1 <: nat)
  {
    star_over_partition_range n f i;
    rewrite
       (star_over_partition f (select (disjoint_partitions_range n) i))
    as (f i ** f (n - i - 1 <: nat))
  };

  bigstar_map
    #0 #_ #0 #(size `div` 2sz)
    #(fun i ->
      star_over_partition
        (fun i -> gpu_pts_to_cell arr #f i (Seq.index s i))
        (select (disjoint_partitions_range size) i))
    #(fun i ->
      gpu_pts_to_cell arr #f i (Seq.index s i) **
      gpu_pts_to_cell arr #f (SZ.v size - i - 1) (index_flip s i))
    (star_over_partition_range
        size
        (fun i -> gpu_pts_to_cell arr #f i (Seq.index s i)));
}

ghost
fn partition_cells_inv
 (#a:Type0)
  (#size:sz { size % 2sz == 0sz })
  (arr : gpu_array a size)
  (#f : perm)
  (#s:seq a { len s == SZ.v size })
requires
  bigstar 0 (SZ.v size / 2) (fun i ->
    gpu_pts_to_cell arr #f i (Seq.index s i) **
    gpu_pts_to_cell arr #f (SZ.v size - i - 1) (index_flip s i))
ensures
  bigstar 0 size (fun (i:idx 0 (SZ.v size)) ->
    gpu_pts_to_cell arr #f i (Seq.index s i))
{
  ghost
  fn star_over_partition_range_inv
      (n:nat { n % 2 == 0 })
      (f:idx 0 n -> slprop)
      (i:nat { 0 <= i /\ i < n / 2 })
  requires
    f i ** f (n - i - 1 <: nat)
  ensures
    star_over_partition f (select (partition_range n) i)
  {
    star_over_partition_range n f i;
    rewrite
        (f i ** f (n - i - 1 <: nat))
     as (star_over_partition f (select (partition_range n) i))
  };

  bigstar_map
    #_ #_ #0 #(SZ.v size / 2)
    #(fun i ->
      gpu_pts_to_cell arr #f i (Seq.index s i) **
      gpu_pts_to_cell arr #f (SZ.v size - i - 1) (index_flip s i))
    #(fun i ->
      star_over_partition
        (fun i -> gpu_pts_to_cell arr #f i (Seq.index s i))
        (select (partition_range size) i))
    (star_over_partition_range_inv
        size
        (fun i -> gpu_pts_to_cell arr #f i (Seq.index s i)));

  bigstar_partition_inv size (SZ.v size / 2)
      (fun i -> gpu_pts_to_cell arr #f i (Seq.index s i))
      (disjoint_partitions_range size);
}

inline_for_extraction
fn read_cell
  (#ty:Type0)
  (#size:erased sz)
  (a:gpu_array ty (SZ.v size))
  (i:sz { i < size })
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
  (i:sz { i < size })
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
  let idx' = (size - idx - 1sz);
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

  forevery_fromstar #(natlt (size `div` 2sz))
    (fun tid ->
      gpu_pts_to_cell a #1.0R tid (Seq.index s tid) **
      gpu_pts_to_cell a #1.0R (SZ.v size - tid - 1) (index_flip s tid));
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
  forevery_tostar #(natlt (size /^ 2sz))
    (fun tid ->
      gpu_pts_to_cell a #1.0R tid (Seq.index (reverse_spec s) tid) **
      gpu_pts_to_cell a #1.0R (SZ.v size - tid - 1) (index_flip (reverse_spec s) tid));

  partition_cells_inv a;
  implode_cells a
}

(* TODO: if size is odd, just put the middle cell in the frame! *)
inline_for_extraction noextract
let kdesc
  (#ty:Type0)
  (size:sz { size > 0sz /\ size % 2sz == 0sz /\ SZ.v size < reveal max_blocks })
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
