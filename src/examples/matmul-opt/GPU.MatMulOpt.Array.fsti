module GPU.MatMulOpt.Array

#push-options "--fuel 1 --ifuel 1"

#lang-pulse

open FStar.Tactics.V2
open FStar.Mul
open Pulse.Lib.Pervasives
open GPU

open GPU.MatMulOpt.Layout

type mseq (dims: FStar.Seq.seq pos) (a: Type) = (s: FStar.Seq.seq a { FStar.Seq.length s == multiply dims })

let init #a (dims: FStar.Seq.seq pos) (f: (idxs: FStar.Seq.seq nat { FStar.Seq.length idxs == FStar.Seq.length dims /\ elementwise_smaller idxs dims }) -> a): mseq dims a
  = FStar.Seq.init (multiply dims) (fun i -> f (split_to_dims dims i))

let index #dims #a (s: mseq dims a) (idx: FStar.Seq.seq nat { FStar.Seq.length dims == FStar.Seq.length idx /\ elementwise_smaller idx dims }): a
  =  s.[join_from_dims dims idx]

let remove #a (s: FStar.Seq.seq a) (idx: nat { idx < FStar.Seq.length s }): (r: FStar.Seq.seq a { FStar.Seq.length r == FStar.Seq.length s - 1 })
  = FStar.Seq.init (FStar.Seq.length s - 1) (fun i -> if i < idx then s.[i] else s.[i + 1])

let insert #a (s: FStar.Seq.seq a) (elem: a) (idx: nat { idx <= FStar.Seq.length s }): (r: FStar.Seq.seq a { FStar.Seq.length r == FStar.Seq.length s + 1 })
  = FStar.Seq.init (FStar.Seq.length s + 1) (fun i -> if i < idx then s.[i] else if i = idx then elem else s.[i - 1])

// TODO
let lemma_ews_remove_insert (dims: FStar.Seq.seq pos) (dim: nat { dim < FStar.Seq.length dims }) (idx: nat { idx < dims.[dim] }) (i: nat { i < multiply (remove dims dim) }):
  Lemma (elementwise_smaller (insert (split_to_dims (remove dims dim) i) idx dim) dims) = admit()

let slice #dims #a (s: mseq dims a) (dim: nat { dim < FStar.Seq.length dims }) (idx: nat { idx < dims.[dim] }): mseq (remove dims dim) a
  = let new_dims = remove dims dim in FStar.Seq.init (multiply new_dims) (fun i ->
    let idxs = insert (split_to_dims new_dims i) idx dim in
      lemma_ews_remove_insert dims dim idx i;
      index s idxs
  )

let gpu_matrix (a:Type u#0) (dims: FStar.Seq.seq pos) : Type u#0 = gpu_array a (multiply dims)

let gpu_pts_to_matrix
  (#a: Type u#0)
  (#dims: FStar.Seq.seq pos)
  (x: gpu_matrix a dims)
  (#[exact (`1.0R)] f : perm)
  (v : mseq dims a)
: slprop = gpu_pts_to_array x #f v

val slice_matrix #a #dims (ga: gpu_matrix a dims) (dim: nat { dim < FStar.Seq.length dims }) (idx: nat { idx < dims.[dim] }): gpu_matrix a (remove dims dim)
  // = ga + idx * multiply dims[..dim]

fn gpu_matrix_read
  #a
  (#dims: FStar.Seq.seq pos)
  (ga : gpu_matrix a dims)
  (#s: erased (mseq dims a))
  (idxs: FStar.Seq.seq nat { FStar.Seq.length dims == FStar.Seq.length idxs /\ elementwise_smaller idxs dims })
  requires gpu ** gpu_pts_to_matrix ga #1.0R s
  returns v: a
  ensures gpu ** gpu_pts_to_matrix ga #1.0R s ** pure (v == index s idxs)

ghost
fn gpu_matrix_slice_permission
  #a
  (#dims: FStar.Seq.seq pos)
  (ga : gpu_matrix a dims)
  (#f : perm)
  (#s: erased (mseq dims a))
  (dim: nat { dim < FStar.Seq.length dims })
  requires gpu ** gpu_pts_to_matrix ga #f s
  ensures gpu ** bigstar 0 dims.[dim] (fun i -> gpu_pts_to_matrix (slice_matrix ga dim i) #f (slice s dim i))
