module GPU.ArrayReversal
#lang-pulse
open FStar.Tactics
open Pulse.Lib
open Pulse.Lib.Pervasives
open GPU
open Pulse.Lib.BoundedIntegers
open FStar.SizeT
module SZ = FStar.SizeT
let named (n:string) (x:slprop) = x
fn intro_named (n:string) (x:slprop)
requires x
ensures named n x
{
  fold (named n x)
}

open FStar.OrdSet
let idx (m n:nat) = i:nat { m <= i /\ i < n }
let idx_set (m n:nat) = FStar.OrdSet.ordset (idx m n) (fun (x y:nat) -> x <= y)

let partitions (m:nat) (n:nat) = seq (idx_set m n)
let union_partitions #m #n (p:partitions m n) = FStar.Seq.foldr OrdSet.union p OrdSet.empty  
let range (m:nat) (n:nat { m <= n }) : s:idx_set m n { forall i. OrdSet.mem i s <==> m <= i /\ i < n } =
  let rec aux (k:nat { m <= k /\ k <= n }) 
  : Tot (s:idx_set m n { forall i. OrdSet.mem i s <==> k <= i /\ i < n}) (decreases (n - k))
  = if k = n then OrdSet.empty
    else OrdSet.union (OrdSet.singleton k) (aux (k + 1 <: nat))
  in
  aux m

let disjoint_partitions (m:nat) (n:nat) =
  l:seq (idx_set m n) {
    m <= n  /\
    (forall (i j:nat). {:pattern (Seq.index l i); (Seq.index l j)}
       i < j /\ j < Seq.length l ==> OrdSet.disjoint (Seq.index l i) (Seq.index l j)) /\
    (union_partitions l `OrdSet.equal` range m n)
  }

let star_over_partition (m:nat) (n:nat{m<n}) (f:idx m n -> slprop) (partition:idx_set m n)
: slprop
= f m

 
// ghost
// fn bigstar_rewrite
//   (m0:nat)
//   (n0:nat{m0 <= n0})
//   (m1:nat)
//   (n1:nat{m1 <= n1})
//   (f0: (i:nat { m0 <= i /\ i < n0 } -> slprop))
//   (f1: (i:nat { m1 <= i /\ i < n1 } -> slprop))
//   (partition: disjoint_partitions m0 n0)
// requires
//   bigstar m0 n0 f0
//   pure (forall (i:nat { m1 <= i /\ i < n1 }). f1 i == f0 i)
// ensures
//   bigstar m1 n1 f1
// {
//   admit()
// }

let gpu_pts_to_cell
  (#a:Type0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
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

fn read_cell_named
  (#ty:Type0)
  (#size:erased sz)
  (a:gpu_array ty (SZ.v size))
  (i:sz { i < size })
  (#name:string)
  (#u:erased ty)
requires
  gpu **
  named name (gpu_pts_to_cell a (SZ.v i) u)
returns v:ty
ensures
  gpu **
  named name (gpu_pts_to_cell a (SZ.v i) u) **
  pure (u == v)
{
  unfold named;
  unfold (gpu_pts_to_cell a (SZ.v i) u);
  //Why do I have to instantiate the implicits?
  let v = gpu_array_read #_ #_ #(SZ.v i) #(SZ.v i + 1) a i;
  fold (gpu_pts_to_cell a (SZ.v i) u);
  fold (named name (gpu_pts_to_cell a (SZ.v i) u));
  v
}

fn write_cell_named
  (#ty:Type0)
  (#size:erased sz)
  (a:gpu_array ty (SZ.v size))
  (i:sz { i < size })
  (v:ty)
  (#name:string)
  (#u:erased ty)
requires
  gpu **
  named name (gpu_pts_to_cell a (SZ.v i) u)
ensures
  gpu **
  named name (gpu_pts_to_cell a (SZ.v i) v)
{
  unfold named;
  unfold (gpu_pts_to_cell a (SZ.v i) u);
  //Why do I have to instantiate the implicits?
  gpu_array_write #_ #_ #(SZ.v i) #(SZ.v i + 1) a i v;
  fold (gpu_pts_to_cell a (SZ.v i) v);
  fold (named name (gpu_pts_to_cell a (SZ.v i) v))
}


[@@CPrologue "__global__"]
fn kernel
  (#ty:Type0)
  (size:sz)
  (a:gpu_array ty size)
  (etid:erased tid_t { bidx_x etid < size `div` 2sz /\ gdim_x etid == 1sz })
  (#u #v:erased ty)
requires
  gpu **
  thread_id etid **
  named "cell_l" (gpu_pts_to_cell a (bidx_x etid) u) **
  named "cell_r" (gpu_pts_to_cell a (size - bidx_x etid - 1sz) v)
ensures
  gpu **
  thread_id etid **
  named "cell_l" (gpu_pts_to_cell a (bidx_x etid) v) **
  named "cell_r" (gpu_pts_to_cell a (size - bidx_x etid - 1sz) u)
{
  let tid = GPU.Base.block_idx_x ();
  let idx = uint32_to_sizet tid;
  let idx' = (size - idx - 1sz);
  let uu = read_cell_named a idx #"cell_l";
  let vv = read_cell_named a idx' #"cell_r";
  write_cell_named a idx vv #"cell_l";
  write_cell_named a idx' uu #"cell_r"
}

fn reverse
    (#ty:Type0)
    (size:sz { size % 2sz == 0sz /\ SZ.v size < max_blocks })
    (a:gpu_array ty size)
    (#s:FStar.Seq.seq ty { Seq.length s == SZ.v size })
requires
  cpu **
  gpu_pts_to_array a s
ensures
  cpu **
  gpu_pts_to_array a s
{
  explode_cells a;
  admit();
  ()
}
