module Kuiper.Kernel.GEMMCPU

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
open Kuiper.EMatrix { ematrix, to_real_matrix }

#set-options "--z3rlimit 20"

inline_for_extraction noextract
fn copy_from_vec
  (#et:Type0) {| sized et |}
  (#rows #cols : sz)
  (#l : layout2 rows cols { is_full l })
  (gm : array2 et l)
  (a : vec et)
  (#s : erased (seq et){Seq.length s == rows * cols})
  (#em : ematrix et rows cols)
  preserves
    cpu ** a |-> s
  requires
    on gpu_loc (gm |-> em)
  ensures
    on gpu_loc (gm |-> from_seq l s)
{
  map_loc gpu_loc
    #(gm |-> em)
    #(core gm |-> to_seq l em)
    fn () { tensor_concr gm; };
  Pulse.Lib.Vec.pts_to_len a;
  gpu_memcpy_host_to_device (core gm) a (rows *^ cols);
  map_loc gpu_loc
    #(core gm |-> s)
    #(gm |-> from_seq l s)
    fn () {
      tensor_abs' l (core gm);
      rewrite from_array l (core gm) |-> Frac 1.0R (from_seq l s)
           as gm |-> from_seq l s;
    };
}

inline_for_extraction noextract
fn copy_to_vec
  (#et:Type0) {| sized et |}
  (#rows #cols : sz)
  (#l : layout2 rows cols { is_full l })
  (a : vec et)
  (gm : array2 et l)
  (#s : erased (seq et){Seq.length s == rows * cols})
  (#em : ematrix et rows cols)
  preserves
    cpu ** on gpu_loc (gm |-> em)
  requires
    a |-> s
  ensures
    a |-> to_seq l em
{
  map_loc gpu_loc
    #(gm |-> em)
    #(gm |-> em ** pure (SZ.fits (l.ulen)))
    fn () { tensor_pts_to_ref gm; };
  Pulse.Lib.Vec.pts_to_len a;
  map_loc gpu_loc
    #(gm |-> em)
    #(core gm |-> Frac 1.0R (to_seq l em))
    fn () { tensor_concr gm; };
  gpu_memcpy_device_to_host a (core gm) (rows *^ cols);
  map_loc gpu_loc
    #(core gm |-> Frac 1.0R (to_seq l em))
    #(gm |-> em)
    fn () {
      tensor_abs l (core gm);
      rewrite from_array l (core gm) |-> Frac 1.0R em
           as gm |-> em;
    };
}

inline_for_extraction noextract
fn matmul_cpu
  (#size_req : size_req_t)
  (mmcomb_gpu : matmulcomb_gpu_ty size_req)
  (#et : Type0) {| scalar et |}
  (#m #n #k : szp) (* concrete args *)
  (#lA : full_layout2 m k)
  (#lB : full_layout2 k n)
  (#lC : full_layout2 m n)
  {| ctlayout lA, ctlayout lB, ctlayout lC |}
  (a b : vec et)
  (#sa : erased (seq et){len sa == m * k})
  (#sb : erased (seq et){len sb == k * n})
  norewrite
  preserves
    cpu ** a |-> sa ** b |-> sb
  requires
    pure (size_req m n k)
  returns
    c : vec et
  ensures
    c |-> (to_seq lC <|
             MS.matmul (from_seq lA sa)
                       (from_seq lB sb))
{
  let gA = alloc0 #et (m *^ k) lA;
  let gB = alloc0 #et (k *^ n) lB;
  let gC = alloc0 #et (m *^ n) lC;

  copy_from_vec gA a;
  copy_from_vec gB b;

  mmcomb_gpu MS.comb2 gA gB gC;

  let c = Pulse.Lib.Vec.alloc #et zero (SZ.mul m n);
  copy_to_vec c gC;

  free gA;
  free gB;
  free gC;

  c
}

(* This will dinamically abort if the dimensions (rows/shared/cols) are not
   multiples of tile. *)
inline_for_extraction noextract
fn mmcomb_gpu_tiled
  (#size_req : tiled_size_req_t)
  (mmcomb_gpu : tiled_matmulcomb_gpu_ty size_req)
  (tile : valid_tile)
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| cA : ctlayout lA, cB : ctlayout lB, cC : ctlayout lC |}
  (gA : array2 et lA { is_global gA })
  (gB : array2 et lB { is_global gB })
  (gC : array2 et lC { is_global gC })
  (#eA #eB #eC : ematrix et _ _)
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (size_req (m / tile) (n / tile) (k / tile) tile) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)
{
  dassert (tile >^ 0sz);
  dguard (m %^ tile = 0sz);
  dguard (n %^ tile = 0sz);
  dguard (k %^ tile = 0sz);
  let mm = m /^ tile;
  let nn = n /^ tile;
  let kk = k /^ tile;

  // None of these implicits should be needed. (Well, maybe the first
  // three until Kuiper.Concrete works really well.)
  mmcomb_gpu tile comb
    #mm #nn #kk
    #_ #_ #_
    #cA #cB #cC
    gA gB gC;

  ()
}
