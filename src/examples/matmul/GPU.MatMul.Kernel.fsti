module GPU.MatMul.Kernel

#push-options "--fuel 1 --ifuel 1"

open FStar.Mul
open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open GPU
module Impure = GPU.MatMul.Impure
module Pure = GPU.MatMul.Pure
module SZ = FStar.SizeT

let singleton #a (elem: a) : Seq.Base.seq a = Seq.Base.cons elem Seq.Base.empty

let kpre_pair (rows shared columns: nat)
  (ga1: gpu_array int (rows * shared))
  (ga2: gpu_array int (shared * columns))
  (#s1: erased (Seq.Base.seq int) {Seq.Base.length s1 == rows * shared})
  (#s2: erased (Seq.Base.seq int) {Seq.Base.length s2 == shared * columns})
  (nth: erased nat { nth > 0 })
  : slprop
  =
  Impure.gpu_pts_to_matrix rows shared ga1 nth s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 nth s2

let kpre (rows shared columns: nat)
  (ga1: gpu_array int (rows * shared))
  (ga2: gpu_array int (shared * columns))
  (r: gpu_array int (rows * columns))
  (#s1: erased (Seq.Base.seq int) {Seq.Base.length s1 == rows * shared})
  (#s2: erased (Seq.Base.seq int) {Seq.Base.length s2 == shared * columns})
  (#sr: erased (Seq.Base.seq int) {Seq.Base.length sr == 1})
  (nth: erased nat)
  (tid : nat{tid < nth /\ tid < rows * columns})
  : slprop
  =
  kpre_pair rows shared columns ga1 ga2 #s1 #s2 nth
  ** gpu_pts_to_array_slice r tid (tid+1) sr

let kpost (rows shared columns: nat)
  (ga1: gpu_array int (rows * shared))
  (ga2: gpu_array int (shared * columns))
  (r: gpu_array int (rows * columns))
  (#s1: erased (Seq.Base.seq int) {Seq.Base.length s1 == rows * shared})
  (#s2: erased (Seq.Base.seq int) {Seq.Base.length s2 == shared * columns})
  (nth: erased nat)
  (tid : nat{tid < nth /\ tid < rows * columns})
  : slprop
  =
  Impure.gpu_pts_to_matrix rows shared ga1 nth s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 nth s2
  // ** gpu_pts_to_array_slice r tid (tid+1) (singleton (reveal (matmul_single rows shared columns s1 s2 (tid / columns) (tid % columns) shared)))
  ** (exists* s. gpu_pts_to_array_slice r tid (tid+1) s)

// TODO: un-hardcode
let rows : nat = 1024 // rows of ga1/r
// assume val rows : nat
let shared : nat = 1024 // columns of ga1, rows of ga2
let columns : nat = 1024 // columns of ga2/r

```pulse
fn kernel
  // (rows: nat) (shared: nat { shared < pow2 16 }) (columns: nat)
  (ga1 : gpu_array int (rows * shared)) (ga2 : gpu_array int (shared * columns)) (r : gpu_array int (rows * columns))
  (#s1: erased (Seq.Base.seq int) {Seq.Base.length s1 == rows * shared})
  (#s2: erased (Seq.Base.seq int) {Seq.Base.length s2 == shared * columns})
  (#sr: erased (Seq.Base.seq int) {Seq.Base.length sr == 1})
  (nth : erased nat)
  (tid : nat{tid < nth /\ tid < rows * columns})
  requires gpu
    ** kpre rows shared columns ga1 ga2 r #s1 #s2 #sr nth tid
  ensures  gpu
    ** kpost rows shared columns ga1 ga2 r #s1 #s2 nth tid
{
  unfold kpre rows shared columns ga1 ga2 r #s1 #s2 #sr nth tid;
  unfold kpre_pair rows shared columns ga1 ga2 #s1 #s2 nth;

  (* r[tid] = TODO *)
  let trow = tid / columns;
  let tcol = tid % columns;
  // assert (pure (0 <= trow /\ trow < rows /\ 0 <= tcol /\ tcol < columns));

  let mut i = 0sz;
  let mut sum = 0;

  while (let v = !i; (SZ.v v < shared))
     invariant b.
       exists* v.
       pure (0 <= shared /\ b == (SZ.v v < shared) /\ SZ.v v <= shared) **
       pts_to i v **
       gpu **
       pts_to sum (Pure.matmul_single rows shared columns s1 s2 trow tcol (SZ.v v))
       ** Impure.gpu_pts_to_matrix rows shared ga1 nth s1
       ** Impure.gpu_pts_to_matrix shared columns ga2 nth s2
  {
    let v = !i;
    let s = !sum;
    let v1 = Impure.gpu_matrix_read #int #rows #shared ga1 #nth #s1 trow (SZ.v v);
    let v2 = Impure.gpu_matrix_read #int #shared #columns ga2 #nth #s2 (SZ.v v) tcol;

    i := SZ.add v 1sz;
    sum := s + (v1 * v2);

    (**)Pure.matmul_single_lemma rows shared columns s1 s2 trow tcol (SZ.v (SZ.add v 1sz));
    ()
  };

  let s = !sum;
  gpu_array_write #int #_ #tid #(tid+1) r tid s;
  (**)Seq.Base.lemma_eq_intro (Seq.Base.upd sr 0 s) (singleton s);

  fold kpost rows shared columns ga1 ga2 r #s1 #s2 nth tid;
  ()
}
```


```pulse
ghost fn fold_pre_pair
  (rows shared columns: nat)
  (ga1: gpu_array int (rows * shared))
  (ga2: gpu_array int (shared * columns))
  (#s1: erased (Seq.Base.seq int) {Seq.Base.length s1 == rows * shared})
  (#s2: erased (Seq.Base.seq int) {Seq.Base.length s2 == shared * columns})
  (nth: erased nat { nth > 0 })
  (tid: nat)
  requires Impure.gpu_pts_to_matrix rows shared ga1 nth s1
        ** Impure.gpu_pts_to_matrix shared columns ga2 nth s2
  ensures  kpre_pair rows shared columns ga1 ga2 #s1 #s2 nth
{
  fold kpre_pair rows shared columns ga1 ga2 #s1 #s2 nth;
  ()
}
```

```pulse
ghost fn fold_pre
  (rows shared columns: nat)
  (ga1: gpu_array int (rows * shared))
  (ga2: gpu_array int (shared * columns))
  (gr: gpu_array int (rows * columns))
  (#s1: erased (Seq.Base.seq int) {Seq.Base.length s1 == rows * shared})
  (#s2: erased (Seq.Base.seq int) {Seq.Base.length s2 == shared * columns})
  (#sr: (Seq.Base.seq int) {Seq.Base.length sr == 1})
  (nth: erased nat)
  (tid: nat { tid < nth /\ tid < rows * columns })
  requires kpre_pair rows shared columns ga1 ga2 #s1 #s2 nth
        ** gpu_pts_to_array_slice gr tid (tid+1) sr
  ensures  kpre rows shared columns ga1 ga2 gr #s1 #s2 #sr nth tid
{
  fold kpre rows shared columns ga1 ga2 gr #s1 #s2 #sr nth tid;
  ()
}
```

#pop-options
