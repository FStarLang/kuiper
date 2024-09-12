module GPU.ArrayReversal
#lang-pulse
open FStar.Tactics
open Pulse.Lib
open Pulse.Lib.Pervasives
open GPU
open Pulse.Lib.BoundedIntegers
open Pulse.Lib.PartitionRange
open FStar.SizeT
open FStar.FiniteSet.Base
open FStar.FiniteSet.Ambient
module Set = FStar.FiniteSet.Base
module SZ = FStar.SizeT

noextract
let index_flip (#a:Type) (s:seq a) (i:nat { i < Seq.length s }) = Seq.index s (Seq.length s - i - 1 <: nat)
noextract
let reverse_spec (#a:Type) (s:seq a) = Seq.init (Seq.length s) (fun i -> index_flip s i)

noextract
let partition_range (n:nat { n % 2 == 0 })
: partitions 0 n (n / 2)
= fun i -> Set.union (Set.singleton i) (Set.singleton (n - i - 1))

noextract
let star_over_partition_range (n:nat { n % 2 == 0 }) (f:idx 0 n -> slprop) (i:nat { i < n / 2 })
: Lemma
  (ensures star_over_partition f (select (partition_range n) i) ==
            f i ** f (n - i - 1 <: nat))
= slprop_equivs() 

noextract
let partition_range_disjoint (n:nat { n % 2 == 0 })
: Lemma (parts_disjoint (partition_range n))
= ()

#restart-solver
#push-options "--query_stats --fuel 0 --ifuel 0"
let partition_covers_range_lemma (n:nat { n % 2 == 0 })
: Lemma (ensures parts_covers_range (partition_range n))
= introduce forall z. z `Set.mem` range 0 n ==> z `Set.mem` union_partitions (partition_range n)
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
  (arr : gpu_array a sz)
  (#f : perm)
  (i:nat)
  (v:a)
: slprop
= exists* s. gpu_pts_to_array_slice arr #f i (i+1) s **
             pure (s `Seq.equal` Seq.create 1 v)

ghost
fn explode_cells
  (#a:Type0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#s:seq a { Seq.length s == sz })
requires
  gpu_pts_to_array arr #f s
ensures
  bigstar 0 sz (fun i -> gpu_pts_to_cell arr #f i (Seq.index s i))
{
  gpu_array_slice_1 arr;
  ghost
  fn slice_to_cell (i:nat { 0 <= i /\ i < sz })
  requires
    gpu_pts_to_array_slice arr #f i (i+1) seq![Seq.index s i]
  ensures
    gpu_pts_to_cell arr #f i (Seq.index s i)
  {
    fold (gpu_pts_to_cell arr #f i (Seq.index s i))
  };
  bigstar_map
    #_ #_ #0 #sz
    #(fun i -> gpu_pts_to_array_slice arr #f i (i+1) seq![Seq.index s i])
    #(fun i -> gpu_pts_to_cell arr #f i (Seq.index s i))
    slice_to_cell;
}


ghost
fn implode_cells
  (#a:Type0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#s:seq a { Seq.length s == sz })
requires
  bigstar 0 sz (fun (i:idx 0 sz) -> gpu_pts_to_cell arr #f i (Seq.index s i))
ensures
  gpu_pts_to_array arr #f s
{
  ghost
  fn cell_to_slice (i:nat { 0 <= i /\ i < sz })
  requires
    gpu_pts_to_cell arr #f i (Seq.index s i)
  ensures
    gpu_pts_to_array_slice arr #f i (Prims.op_Addition i 1) seq![Seq.index s i]
  {
    unfold (gpu_pts_to_cell arr #f i (Seq.index s i));
    with s'. assert (gpu_pts_to_array_slice arr #f i (i+1) s');
    assert pure (reveal s' `Seq.equal` seq![Seq.index (reveal s) i]);
  };
  bigstar_map
    #_ #_ #0 #sz
    #(fun (i:idx 0 sz) -> gpu_pts_to_cell arr #f i (Seq.index s i))
    #(fun i -> gpu_pts_to_array_slice arr #f i (Prims.op_Addition i 1) seq![Seq.index s i])
    cell_to_slice;
  gpu_array_unslice_1 arr #f #s
}

ghost
fn partition_cells
  (#a:Type0)
  (#size:sz { size % 2sz == 0sz })
  (arr : gpu_array a size)
  (#f : perm)
  (#s:seq a { Seq.length s == SZ.v size })
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
  (#s:seq a { Seq.length s == SZ.v size })
requires
  bigstar 0 (SZ.v (size `div` 2sz)) (fun i ->
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
    #_ #_ #0 #(size `div` 2sz)
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

  bigstar_partition_inv size (size `div` 2sz)
      (fun i -> gpu_pts_to_cell arr #f i (Seq.index s i))
      (disjoint_partitions_range size);


}

[@@CPrologue "__device__"]
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
  unfold gpu_pts_to_cell; //a (SZ.v i) u);
  //Why do I have to instantiate the implicits?
  let v = gpu_array_read #_ #_ #(SZ.v i) #(SZ.v i + 1) a i;
  fold (gpu_pts_to_cell a #p (SZ.v i) u);
  v
}

[@@CPrologue "__device__"]
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
  //Why do I have to instantiate the implicits?
  gpu_array_write #_ #_ #(SZ.v i) #(SZ.v i + 1) a i v;
  fold (gpu_pts_to_cell a #1.0R (SZ.v i) v);
}


[@@CPrologue "__global__"]
fn kernel
  (#ty:Type0)
  (size:sz)
  (a:gpu_array ty size)
  (#s:erased (Seq.seq ty) { Seq.length s == SZ.v size })
  (etid:tid_t { gdim_x etid == size `div` 2sz /\ bdim_x etid == 1sz }) //thread_index etid < SZ.v size / 2 })
requires
  gpu **
  thread_id etid **
  (gpu_pts_to_cell a #1.0R (thread_index etid) (Seq.index s (thread_index etid)) **
   gpu_pts_to_cell a #1.0R (SZ.v size - thread_index etid - 1) (index_flip s (thread_index etid)))
ensures
  gpu **
  thread_id etid **
  (gpu_pts_to_cell a #1.0R (thread_index etid) (Seq.index (reverse_spec s) (thread_index etid)) **
   gpu_pts_to_cell a #1.0R (SZ.v size - thread_index etid - 1sz) (index_flip (reverse_spec s) (thread_index etid)))
{
  let idx = GPU.Base.thread_idx_all ();
  let idx' = (size - idx - 1sz);
  let uu = read_cell a idx;
  let vv = read_cell a idx';
  write_cell a idx vv;
  write_cell a idx' uu;
}

fn reverse
    (#ty:Type0)
    (size:sz { size > 0sz /\ size % 2sz == 0sz /\ SZ.v size < max_blocks })
    (a:gpu_array ty size)
    (#s: erased (FStar.Seq.seq ty) { Seq.length s == SZ.v size })
requires
  cpu **
  gpu_pts_to_array a s
ensures
  cpu **
  gpu_pts_to_array a (reverse_spec s)
{
  explode_cells a;
  partition_cells a;
  launch_kernel_n
    (size `div` 2sz) 
    #(fun tid -> 
      gpu_pts_to_cell a #1.0R tid (Seq.index s tid) **
      gpu_pts_to_cell a #1.0R (SZ.v size - tid - 1) (index_flip s tid))
    #(fun tid -> 
      gpu_pts_to_cell a #1.0R tid (Seq.index (reverse_spec s) tid) **
      gpu_pts_to_cell a #1.0R (SZ.v size - tid - 1) (index_flip (reverse_spec s) tid))
    (fun etid -> kernel size a #s etid);
  partition_cells_inv a;
  implode_cells a
}

// GM: suprised this worked
let reverse_u64 = reverse #u64
