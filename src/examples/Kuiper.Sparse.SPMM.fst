module Kuiper.Sparse.SPMM

//#set-options "--z3rlimit 20"
#set-options "--debug SMTFail --split_queries always"

#lang-pulse

open Kuiper
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
open Kuiper.Sparse
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Reprs
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Poly.GEMMGPU.Type { size_req_t }

type parameters = {
  rows : szp;
  shared : szp;
  cols : szp;
  blockItemsK : szp;
  blockItemsX : szp;
  blockWidth : (k : szp {k /? blockItemsK /\ k /? blockItemsX});
  // TODO quitar esto
  no_mask : squash (blockItemsX /? cols)
}

(* Shadow lseq to make it erased. *)
let lseq (a:Type) (n:nat) = erased (Seq.lseq a n)


inline_for_extraction noextract
let size_req : size_req_t =
  fun mrows mshared mcols ->
    mrows <= max_blocks /\
    mcols <= max_threads

let nblocks (p : parameters) = p.rows * p.cols / p.blockItemsX
let nthreads (p : parameters) = nblocks p * p.blockWidth

let brow (p : parameters) (bid : natlt (nblocks p))
  //= bid / ((p.cols + p.blockItemsX - 1) / p.blockItemsX)
  : GTot nat
  = bid / (p.cols / p.blockItemsX)

unfold
let n_idx (p : parameters) (bid : szlt (nblocks p)) : sz
  = bid %^ (p.cols /^ p.blockItemsX)

let bcol (p : parameters) (bid : natlt (nblocks p))
  // definir en terminos de n_idx??
  //= bid % ((p.cols + p.blockItemsX - 1) / p.blockItemsX)
  = bid % (p.cols / p.blockItemsX)


//let threadItemsX (p : parameters)
  //= p.blockItemsX /^ p.blocWidth

inline_for_extraction noextract
instance sized_sz : sized sz = {
  size = 4sz;
  default = 0sz;
}

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc (et:Type0) {| sized et |}
  (p : parameters) : list shmem_desc = [
  SHArray et p.blockItemsK;
  // TODO podemos parametrizar este tipo?
  // SHArray u32 blockItemsK;
  SHArray sz p.blockItemsK;
]

unfold
let well_formed
  (p : parameters)
  (#nnz : sz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  : prop
  =
  // condicion de smatrix
  valid_smatrix p.rows p.shared (cast_pos col_ind) (cast_pos row_off) /\

  // TODO quitar estas condiciones
  // esta dice que no hay residuo
  (forall (row : natlt p.rows).
    let ri = row_off @! row in
    let re = row_off @! row + 1 in
    p.blockItemsK /? (re - ri))

noextract
let block_lemma whole block k
  : Lemma (requires block /? whole /\ k * block < whole)
          (ensures k * block + block <= whole)
  = ()

noextract
let block_lemma_off whole block k off
  : Lemma (requires block /? whole /\ k * block < whole)
          (ensures off < block ==> k * block + off < whole)
  = ()


//noextract
//let sparse_offset_lemma
  //(nnz : nat)
  //(p : parameters)
  //(idx : nat)
  //(tid : natlt p.blockWidth)
  //: Lemma (requires p.blockItemsK /? nnz /\ idx * p.blockItemsK < nnz)
          //(ensures forall (k : natlt(p.blockItemsK / p.blockWidth)).
            //idx * p.blockItemsK + k * p.blockWidth + tid < nnz)
  //= 
  //introduce forall (k : natlt(p.blockItemsK / p.blockWidth)).
    //idx * p.blockItemsK + k * p.blockWidth + tid < nnz
  //with (
    //block_lemma_off nnz p.blockItemsK idx (k * p.blockWidth + tid)
  //)

noextract
let barrier_p_odd
  (#et : Type0)
  (p : parameters)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (ri re : nat{ri <= re /\ re <= nnz})
  (idx : nat)
  (tid : natlt p.blockWidth)
  (k : natlt (p.blockItemsK /^ p.blockWidth))
  : Ghost slprop
    (requires
      p.blockItemsK /? (re - ri) /\
      ri + idx * p.blockItemsK < re)
    (ensures fun _ -> true)
  = 
  let off = ri + idx * p.blockItemsK in 
  block_lemma_off (re - ri) p.blockItemsK idx (k * p.blockWidth + tid);
  gpu_pts_to_cell elems_tile (k * p.blockWidth + tid)
    (elems @! off + k * p.blockWidth + tid) **
  gpu_pts_to_cell col_ind_tile (k * p.blockWidth + tid)
    (col_ind @! off + k * p.blockWidth + tid)

let barrier_p
  (#et : Type0)
  (p : parameters)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  : B.barrier_side p.blockWidth
  =
  fun it tid ->
    let trow = brow p bid in
    let ri = row_off @! trow in
    let re = row_off @! trow + 1 in
    let off = ri + (it / 2) * p.blockItemsK in 
    // nos pasamos
    if off >= re then emp else (
      if even it then
        (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
        (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
      else (
        forall+ (k : natlt(p.blockItemsK /^ p.blockWidth)).
          barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re (it / 2) tid k
      )
    )
      // Cell elems_tile   (tid <: nat) |-> (elems @! off + tid) **
      // Cell col_ind_tile (tid <: nat) |-> (col_ind @! off + tid)

noextract
let barrier_q_even
  (#et : Type0)
  (p : parameters)
  (nnz : sz)
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (ri re : nat{ri <= re /\ re <= nnz})
  (idx : nat)
  (tid : natlt p.blockWidth)
  (k : natlt(p.blockItemsK /^ p.blockWidth))
  : Ghost slprop
    (requires p.blockItemsK /? (re - ri) /\ ri + idx * p.blockItemsK < re)
    (ensures fun _ -> true)
  = 
  let off = ri + idx * p.blockItemsK in 
  block_lemma_off (re - ri) p.blockItemsK idx (k * p.blockWidth + tid);
  (exists* (x : et). gpu_pts_to_cell elems_tile (k * p.blockWidth + tid) x) **
  (exists* (x : sz). gpu_pts_to_cell col_ind_tile (k * p.blockWidth + tid) x)


let barrier_q
  (#et : Type0)
  (p : parameters)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  : B.barrier_side p.blockWidth
  =
  fun it tid ->
    let trow = brow p bid in
    let ri = row_off @! trow in
    let re = row_off @! trow + 1 in
    let off = ri + (it / 2) * p.blockItemsK in 
    // nos pasamos
    if off >= re then emp else (
      if even it then
        forall+ (k : natlt(p.blockItemsK /^ p.blockWidth)).
          barrier_q_even p nnz elems_tile col_ind_tile ri re (it / 2) tid k
      else (
        block_lemma (re - ri) p.blockItemsK (it / 2);
        elems_tile |-> Frac (1.0R /. p.blockWidth) (Seq.slice elems off (off + p.blockItemsK)) **
        col_ind_tile |-> Frac (1.0R /. p.blockWidth) (Seq.slice col_ind off (off + p.blockItemsK)) 
      )
    )

unfold
let block_pre
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (#lB : mlayout p.shared p.cols)
  (#lC : mlayout p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (eB : ematrix et p.shared p.cols)
  (fA fB : perm)
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  smatrix_pts_to' gA #(fA /. (nthreads p))
    elems col_ind row_off eA **
  gB |-> Frac (fB /. (nthreads p)) eB **
  forall+ (k : nat{k * p.blockWidth + tid < p.blockItemsX}).
    exists* c.
      M.gpu_matrix_pts_to_cell gC
        (brow p bid) (bcol p bid + k * p.blockWidth + tid) c
  

unfold
let block_post
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (#lB : mlayout p.shared p.cols)
  (#lC : mlayout p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (eB : ematrix et p.shared p.cols)
  (fA fB : perm)
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  smatrix_pts_to' gA #(fA /. (nthreads p))
    elems col_ind row_off eA **
  gB |-> Frac (fB /. (nthreads p)) eB **
  forall+ (k : nat{k * p.blockWidth + tid < p.blockItemsX}).
    M.gpu_matrix_pts_to_cell gC
      (brow p bid) (bcol p bid + k * p.blockWidth + tid)
      (MS.matmul_single eA eB (brow p bid) (bcol p bid + k * p.blockWidth + tid))

let barrier_contract
  (#et : Type0)
  (p : parameters)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  : B.contract p.blockWidth =
  {
    rin  = barrier_p p elems col_ind row_off elems_tile col_ind_tile bid;
    rout = barrier_q p elems col_ind row_off elems_tile col_ind_tile bid;
  }

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (#lB : mlayout p.shared p.cols)
  (#lC : mlayout p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (eB : ematrix et p.shared p.cols)
  (fA fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc et p))
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  let (elems_tile, (col_ind_tile, _)) = sh in
  block_pre p gA gB gC elems col_ind row_off eA eB fA fB bid tid **
  (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
  (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (#lB : mlayout p.shared p.cols)
  (#lC : mlayout p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (eB : ematrix et p.shared p.cols)
  (fA fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc et p))
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  let (elems_tile, (col_ind_tile, _)) = sh in
  let trow = brow p bid in
  let ri = row_off @! trow in
  let re = row_off @! trow + 1 in
  block_post p gA gB gC elems col_ind row_off eA eB fA fB bid tid **
  live elems_tile #(1.0R /. p.blockWidth) **
  live col_ind_tile #(1.0R /. p.blockWidth)

ghost
fn barrier_p_fold_even
  (#et : Type0)
  (p : parameters)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  (idx : nat)
  (tid : natlt p.blockWidth)
  (ri : sz{ri == row_off @! brow p bid})
  (re : sz{re == row_off @! brow p bid + 1})
  requires
    pure (ri + idx * p.blockItemsK < re) **
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
  ensures barrier_p p elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid
{
  rewrite (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
          (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
       as barrier_p p elems col_ind row_off elems_tile col_ind_tile bid (idx * 2) tid;
  ();
}

ghost
fn barrier_p_fold_odd
  (#et : Type0)
  (p : parameters)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  (idx : nat)
  (tid : natlt p.blockWidth)
  (ri : sz{ri == row_off @! brow p bid})
  (re : sz{re == row_off @! brow p bid + 1})
  (#_ : squash (ri + idx * p.blockItemsK < re))
  requires
    forall+ (k: natlt (p.blockItemsK /^ p.blockWidth)).
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
  ensures
    barrier_p p elems col_ind row_off elems_tile col_ind_tile bid (idx * 2 + 1) tid
{
  assert pure ((idx * 2 + 1) / 2 == idx);
  assert pure ((row_off @! brow p bid) + ((idx * 2 + 1) / 2) * p.blockItemsK < re);
  assert pure (odd (idx * 2 + 1));
  assert pure (not (even (idx * 2 + 1)));
  // rewrite forall+ (k: natlt (p.blockItemsK /^ p.blockWidth)).
  //           barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
  //      as barrier_p p elems col_ind row_off elems_tile col_ind_tile bid (idx * 2 + 1) tid;
  // ();
  admit();
}

ghost
fn barrier_q_unfold_even
  (#et : Type0)
  (p : parameters)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  (idx : nat)
  (tid : natlt p.blockWidth)
  (ri : sz{ri == row_off @! brow p bid})
  (re : sz{re == row_off @! brow p bid + 1})
  (#_ : squash (ri + idx * p.blockItemsK < re))
  requires
    barrier_q p elems col_ind row_off elems_tile col_ind_tile bid (idx * 2) tid
  ensures
    (forall+ (k : natlt (p.blockItemsK /^ p.blockWidth)).
      barrier_q_even p nnz elems_tile col_ind_tile ri re idx tid k)
  {
    admit();
    //let trow = brow p bid;
    //let ri = row_off @! trow;
    //let re = row_off @! trow + 1;
    //let off = ri + (it / 2) * p.blockItemsK;
    //assert pure (off < re);
    //rewrite
      //barrier_q p elems col_ind row_off elems_tile col_ind_tile
        //bid it tid
      //as forall+ (k : natlt(p.blockItemsK / p.blockWidth)).
          //(exists* (x : et). gpu_pts_to_cell elems_tile (k * p.blockWidth + tid) x) **
          //(exists* (x : sz). gpu_pts_to_cell col_ind_tile (k * p.blockWidth + tid) x)

}
  
ghost
fn barrier_q_unfold_odd
  (#et : Type0)
  (p : parameters)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  (idx : nat)
  (tid : natlt p.blockWidth)
  (ri : sz{ri == row_off @! brow p bid})
  (re : sz{re == row_off @! brow p bid + 1})
  //(#_ : squash (ri + idx * p.blockItemsK < re))
  (#_ : squash (ri + idx * p.blockItemsK + p.blockItemsK <= re))
  requires
    barrier_q p elems col_ind row_off elems_tile col_ind_tile
      bid (idx * 2 + 1) tid
  ensures
    elems_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice elems
        (ri + idx * p.blockItemsK)
        (ri + idx * p.blockItemsK + p.blockItemsK)) ** 
    col_ind_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice col_ind
        (ri + idx * p.blockItemsK)
        (ri + idx * p.blockItemsK + p.blockItemsK))
  {
    admit();
  }


inline_for_extraction noextract
fn foreach
  (n : sz{fits n})
  (p q : natlt n -> slprop)
  (#frame : slprop)
  (f : (i : szlt n) -> stt unit (p i ** frame) (fun _ -> q i ** frame))
  preserves
    frame
  requires
    (forall+ (k : natlt n). p k)
  ensures
    (forall+ (k : natlt n). q k)
{
  let mut k = 0sz;

  forevery_ext p (fun ki -> p (0 + ki));
  forevery_rw_size n (n - 0);

  forevery_intro_false q;
  forevery_refine_ext (fun (ki : natlt n) -> ki < 0) q;

  while ((!k <^ n))
    invariant
      live k ** 
      pure (!k <=^ n) **
      (forall+ (ki : natlt (n - !k)). p (!k + ki)) **
      (forall+ (ki : natlt n{ki < !k}). q ki)
  {
    with vk. assert k |-> vk;

    forevery_natlt_pop_shift (n - vk) (fun ki -> p (vk + ki));
    rewrite p (!k + 0) as p !k;

    f !k;
    k := !k +^ 1sz;
    with vk'. assert k |-> vk';

    forevery_rw_type (natlt (n - vk - 1sz)) (natlt (n - vk')) _;
    forevery_ext
      (fun (ki : natlt (n - vk')) -> p (vk + (ki + 1)))
      (fun (ki : natlt (n - vk')) -> p (vk' + ki));
    
    forevery_insert q vk;
    forevery_refine_ext (fun (ki : natlt n) -> ki < vk') q;

    (); // por que no sale?
    admit();
  };
  with vk. assert k |-> vk;

  forevery_rw_type (natlt (n - vk)) (natlt 0) (fun ki -> p (vk + ki));
  forevery_tonat 0 (fun ki -> p (vk + ki));
  bigstar_zs_elim #_;

  forevery_unrefine q; 
}

inline_for_extraction noextract
fn sparse_load_one
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  // matriz sparse gA
  (#row_off : lseq sz (p.rows + 1))
  (#elems : lseq et gA.nnz)
  (#col_ind : lseq sz gA.nnz)
  (#eA : ematrix et p.rows p.shared)
  (#fA : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (bid : szlt (nblocks p))
  (tid : szlt (p.blockWidth))
  (idx : sz)
  (ri re : sz{ri < re /\ re <= gA.nnz})
  (#_ : squash (p.blockItemsK /? (re - ri)))
  (#_ : squash (ri + idx * p.blockItemsK < re))
  (k : szlt (p.blockItemsK /^ p.blockWidth))
  requires
    barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid k **
    gpu ** 
    smatrix_pts_to' gA #fA elems col_ind row_off eA
  ensures
    barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k **
    gpu ** 
    smatrix_pts_to' gA #fA elems col_ind row_off eA
{
  let tile_off = k *^ p.blockWidth +^ tid;
  assert rewrites_to tile_off (k *^ p.blockWidth +^ tid);

  let off = ri +^ idx *^ p.blockItemsK;
  assert rewrites_to off (ri +^ idx *^ p.blockItemsK);

  unfold barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid k;
  block_lemma_off (re - ri) p.blockItemsK idx tile_off;

  let x = gpu_array_read gA.elems (off +^ tile_off);
  gpu_array_write elems_tile tile_off x;
  with s. assert gpu_pts_to_slice elems_tile tile_off (tile_off + 1) s;
  assert pure (Seq.equal s seq![elems @! off +^ tile_off]);
  assert gpu_pts_to_cell elems_tile tile_off
      (elems @! off +^ tile_off);

  let c = gpu_array_read gA.col_ind (off +^ tile_off);
  gpu_array_write col_ind_tile tile_off c;
  with s. assert gpu_pts_to_slice col_ind_tile tile_off (tile_off + 1) s;
  assert pure (Seq.equal s seq![col_ind @! off +^ tile_off]);
  assert gpu_pts_to_cell col_ind_tile tile_off 
    (col_ind @! off +^ tile_off);

  rewrite 
    gpu_pts_to_cell elems_tile   tile_off (elems   @! off +^ tile_off) **
    gpu_pts_to_cell col_ind_tile tile_off (col_ind @! off +^ tile_off)
    as barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k;
}

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn sparse_load
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  // matriz sparse gA
  (#row_off : lseq sz (p.rows + 1))
  (#elems : lseq et gA.nnz)
  (#col_ind : lseq sz gA.nnz)
  (#eA : ematrix et p.rows p.shared)
  (#fA : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (bid : szlt (nblocks p))
  (tid : szlt (p.blockWidth))
  (idx : sz)
  (ri : sz{ri == row_off @! brow p bid})
  (re : sz{re == row_off @! brow p bid + 1})
  (#_ : squash(ri + (idx + 1) * p.blockItemsK <= gA.nnz))
  norewrite
  preserves
    gpu ** 
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    B.barrier_tok (barrier_contract p elems col_ind row_off elems_tile col_ind_tile bid) **
    thread_id p.blockWidth tid
  requires
    B.barrier_state (idx * 2) **
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s) **
    pure (idx <^ (re -^ ri) /^ p.blockItemsK)
  ensures
    B.barrier_state ((idx + 1) * 2) **
    elems_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice elems
        (ri + idx * p.blockItemsK) (ri + idx * p.blockItemsK + p.blockItemsK)) **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice col_ind (ri + idx * p.blockItemsK) (ri + idx * p.blockItemsK + p.blockItemsK))
{
  let off = ri +^ idx *^ p.blockItemsK;

  barrier_p_fold_even p elems col_ind row_off elems_tile col_ind_tile
    bid idx tid ri re;

  rewrite barrier_p p elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2) tid
       as (barrier_contract p elems col_ind row_off elems_tile col_ind_tile bid).rin
            (idx * 2) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p elems col_ind row_off elems_tile col_ind_tile bid).rout
            (idx * 2) tid
       as barrier_q p elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2) tid;

  barrier_q_unfold_even p elems col_ind row_off elems_tile col_ind_tile
    bid idx tid ri re;
        
  assume pure (p.blockItemsK /? (re - ri));
  foreach (p.blockItemsK /^ p.blockWidth)
    (fun ki -> barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid ki)
    (fun ki -> barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid ki) 
    (sparse_load_one p gA #row_off #elems #col_ind #eA elems_tile col_ind_tile bid tid idx ri re);

  barrier_p_fold_odd p elems col_ind row_off elems_tile col_ind_tile
    bid idx tid ri re;

  rewrite barrier_p p elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2 + 1) tid
       as (barrier_contract p elems col_ind row_off elems_tile col_ind_tile bid).rin
            (idx * 2 + 1) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p elems col_ind row_off elems_tile col_ind_tile bid).rout
            (idx * 2 + 1) tid
       as barrier_q p elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2 + 1) tid;

  barrier_q_unfold_odd p elems col_ind row_off elems_tile col_ind_tile
    bid idx tid ri re;

  ();
}
#pop-options


inline_for_extraction noextract
fn compute
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (#lB : mlayout p.shared p.cols)
  {| clayout lB |}
  (gB : M.gpu_matrix et lB)
  (#fB : perm)
  (out : larray et (p.blockItemsX /^ p.blockWidth))
  // fragmentos sparse
  (#v_elems : lseq et p.blockItemsK)
  (#v_col_ind : lseq sz p.blockItemsK)
  (#_ : squash(forall i. 0 <= v_col_ind @! i /\ v_col_ind @! i < p.shared))
  // matriz densa B
  (#eB : erased (ematrix et p.shared p.cols))
  // resultado parcial
  (#v_out : erased (seq et))
  (bid : szlt (nblocks p))
  (tid : szlt p.blockWidth)
  norewrite
  preserves
    gpu **
    elems_tile |-> Frac (1.0R /. p.blockWidth) v_elems **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth) v_col_ind **
    gB |-> Frac (fB /. nthreads p) eB
  requires
    out |-> v_out
  ensures
    //out |-> v_out + dprod v_elems v_col_ind eB (col := tid)
    live out
{
  Pulse.Lib.Array.pts_to_len out;

  let mut k : sz = 0sz;
  while (!k <^ p.blockItemsK)
    invariant
      // TODO decir algo sobre el producto
      live out ** live k **
      pure (
        !k <= p.blockItemsK
        ///\ (!k < blockItemsK ==> !idx * blockItemsK + !k < re - ri) // hace falta?
      )
  {
    let a = gpu_array_read elems_tile !k;
    let c = gpu_array_read col_ind_tile !k;
    let mut x = 0sz;
    while ((!x <^ p.blockItemsX /^ p.blockWidth))
      invariant
        live out ** live k ** live x **
        pure (
          !k < p.blockItemsK /\
          !x <= p.blockItemsX /^ p.blockWidth
        )
    {
      let n_idx : sz = bid %^ (p.cols /^ p.blockItemsX);
      block_lemma_off p.blockItemsX p.blockWidth !x tid;
      assert pure (n_idx < p.cols /^ p.blockItemsX);
      assert pure (n_idx + !x * p.blockWidth + tid < p.cols);

      let b = M.gpu_matrix_read gB c (n_idx +^ !x *^ p.blockWidth +^ tid);
      with o. unfold Pulse.Lib.Array.pts_to out o;
      admit();
      let c = mask_read out !x;
      mask_write out !x (c `add` (a `mul` b));

      x := !x +^ 1sz;
    };

    k := !k +^ 1sz;
  };
}

// fn foo (r : ref int)
//   requires r |-> 1
//   ensures  r |-> 1
// {
//   assert (exists* v. r |-> v);
//   ();
// }


#set-options "--print_implicits"

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (#lB : mlayout p.shared p.cols)
  (#lC : mlayout p.rows p.cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (#elems : lseq et gA.nnz)
  (#col_ind : lseq sz gA.nnz)
  (#row_off : lseq sz (p.rows + 1))
  (#eA : ematrix et p.rows p.shared)
  // matriz densa gB
  (#eB : ematrix et p.shared p.cols)
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc et p))
  (bid : szlt (nblocks p))
  (tid : szlt p.blockWidth)
  ()
  norewrite
  requires
    gpu **
    kpre p gA gB gC elems col_ind row_off eA eB fA fB sh bid tid **
    thread_id p.blockWidth tid **
    block_id (nblocks p) bid **
    B.barrier_tok (barrier_contract p elems col_ind row_off (fst sh) (fst (snd sh)) bid) **
    B.barrier_state 0
  ensures
    gpu **
    kpost p gA gB gC elems col_ind row_off eA eB fA fB sh bid tid **
    thread_id p.blockWidth tid **
    block_id (nblocks p) bid
{
  // renombrar a brow/bcol
  let trow = bid /^ (p.cols /^ p.blockItemsX);
  let tcol = bid %^ (p.cols /^ p.blockItemsX);
  assert pure (SZ.v trow == brow p bid);
  assert pure (SZ.v tcol == bcol p bid);

  // let m_idx = trow;
  // let n_idx = 0;
  // let blockItemsX = 1;

  let (elems_tile, (col_ind_tile, _)) = sh;

  gpu_pts_to_ref elems_tile;
  gpu_pts_to_ref col_ind_tile;

  let ri = gpu_array_read gA.row_off trow;
  let re = gpu_array_read gA.row_off (trow +^ 1sz);

  //let mut dp : et = zero;
  // let out : larray et (p.blockItemsX /^ p.blockWidth) = magic();
  let mut out = [| zero #et #_; (p.blockItemsX /^ p.blockWidth) |];

  let mut nnz : sz = re -^ ri;
  let mut idx = 0sz;

  assert pure (SZ.fits (re - ri));
  assert pure (SZ.fits ((re - ri) / p.blockItemsK));

  assert pure (ri == row_off @! brow p bid);
  assert pure (re == row_off @! brow p bid + 1);

  while (!nnz >=^ p.blockItemsK)
    invariant
      live out ** // TODO decir algo sobre el producto
      live nnz **
      live idx **
      B.barrier_state (!idx * 2) **
      (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
      (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s) **
      pure (
        !idx <= (re -^ ri) /^ p.blockItemsK /\
        !nnz == re -^ ri -^ !idx *^ p.blockItemsK
      ) 
  {
    assert pure (ri + (!idx + 1) * p.blockItemsK <= gA.nnz);
    assert pure (!idx < (re - ri) / p.blockItemsK);

    sparse_load p gA #row_off #elems #col_ind #eA
      elems_tile col_ind_tile bid tid !idx ri re #();

    compute p elems_tile col_ind_tile gB out bid tid;

    idx := !idx +^ 1sz;
    nnz := !nnz -^ p.blockItemsK;
  };

  //M.gpu_matrix_write_cell gC bid tid !dp;
  let mut x : sz = 0sz;
  while (!x <^ p.blockItemsX /^ p.blockWidth)
    invariant live out 
  {
    open Pulse.Lib.Array;
    Pulse.Lib.Array.pts_to_len out;
    // let c = mask_read out !x;
    let c = out.(!x);
    block_lemma_off p.blockItemsX p.blockWidth !x tid;
    M.gpu_matrix_write_cell gC trow (tcol +^ !x *^ p.blockWidth +^ tid);
    ();
  };

  //assume pure (!dp == MS.matmul_single eA eB bid tid);
  admit();

  drop_ (B.barrier_tok _);
  drop_ (B.barrier_state (!idx * 2));

  rewrite
    block_post p gA gB gC elems col_ind row_off eA eB fA fB bid tid **
    live elems_tile #(1.0R /. p.blockWidth) **
    live col_ind_tile #(1.0R /. p.blockWidth)
    as kpost p gA gB gC elems col_ind row_off eA eB fA fB sh bid tid;
}

(*
ghost
fn setup
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et rows shared)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
  (#eA : ematrix et rows shared)
  // matrices densas
  (#eB : ematrix et shared cols)
  (#fA #fB : perm)
  // TODO por que toma unit?
  ()
  norewrite
  requires
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    gB |-> Frac fB eB **
    live gC
  ensures
    (forall+ (bid : natlt rows) (tid : natlt cols).
      kpre1 gA gB gC elems col_ind row_off eA eB fA fB bid tid) **
      emp
{
  admit()
}

ghost
fn block_setup
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et rows shared)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
  (#eA : ematrix et rows shared)
  // matrices densas
  (#eB : ematrix et shared cols)
  (#fA #fB : perm)
  (#blockItemsK : szp)
  (#_ : squash (well_formed #rows #shared #cols col_ind row_off blockItemsK))
  (sh : c_shmems (shmems_desc et blockItemsK))
  (bid : natlt rows)
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt cols).
      kpre1 gA gB gC elems col_ind row_off eA eB fA fB bid tid)
  ensures
    (forall+ (tid : natlt cols).
      kpre gA gB gC elems col_ind row_off eA eB fA fB sh bid tid) **
      emp
{
  admit()
}

ghost
fn block_teardown
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et rows shared)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
  (#eA : ematrix et rows shared)
  // matrices densas
  (#eB : ematrix et shared cols)
  (#fA #fB : perm)
  (#blockItemsK : szp)
  (#_ : squash (well_formed #rows #shared #cols col_ind row_off blockItemsK))
  (sh : c_shmems (shmems_desc et blockItemsK))
  (bid : natlt rows)
  ()
  norewrite
  requires
    // TODO falta el frame aca??
    (forall+ (tid : natlt cols).
      kpost gA gB gC elems col_ind row_off eA eB fA fB sh bid tid) **
      emp
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt cols).
      kpost1 gA gB gC elems col_ind row_off eA eB fA fB bid tid)
{
  admit();
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et rows shared)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
  (#eA : ematrix et rows shared)
  // matrices densas
  (#eB : ematrix et shared cols)
  (#fA #fB : perm)
  ()
  norewrite
  requires
    // TODO falta el frame aca??
    (forall+ (bid : natlt rows) (tid : natlt cols).
      kpost1 gA gB gC elems col_ind row_off eA eB fA fB bid tid) **
      emp
  ensures
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    gB |-> Frac fB eB **
    gC |-> MS.matmul eA eB
{
  admit();
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : sz)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (gB : M.gpu_matrix et lB{M.is_global_matrix gB})
  (gC : M.gpu_matrix et lC{M.is_global_matrix gC})
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
  (eA : ematrix et rows shared)
  // matrices densas
  (#eB : ematrix et shared cols)
  (#fA #fB : perm)
  (blockItemsK : szp)
  (#_ : size_req rows shared cols)
  (#_ : squash (well_formed #rows #shared #cols col_ind row_off blockItemsK))
  : kernel_desc (
      smatrix_pts_to' gA #fA elems col_ind row_off eA **
      gB |-> Frac fB eB **
      live gC
    )
    (
      smatrix_pts_to' gA #fA elems col_ind row_off eA **
      gB |-> Frac fB eB **
      gC |-> MS.matmul eA eB
    )
= {
  nblk = rows;
  nthr = cols;

  barrier_contract = (fun bid ptrs ->
    barrier_contract #et #rows #shared #cols
      elems col_ind row_off (fst ptrs) (fst (snd ptrs)) bid);
  barrier_ok = (fun bid ptrs -> magic());

  shmems_desc = shmems_desc et blockItemsK;

  frame = emp;

  block_pre  = (fun bid -> forall+ (tid : natlt cols).
    kpre1 gA gB gC elems col_ind row_off eA eB fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt cols).
    kpost1 gA gB gC elems col_ind row_off eA eB fA fB bid tid);
  setup    = setup    gA gB gC elems col_ind row_off #_ #_ #fA;
  teardown = teardown gA gB gC elems col_ind row_off #_ #_ #fA;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    gA gB gC elems col_ind row_off #eA #eB #fA #fB;
  block_teardown = block_teardown gA gB gC elems col_ind row_off #eA #eB #fA #fB;

  kpre = kpre gA gB gC elems col_ind row_off eA eB fA fB;
  kpost = kpost gA gB gC elems col_ind row_off eA eB fA fB;

  f = kf gA gB gC;

  block_pre_sendable=solve;
  block_post_sendable=solve;
  kpre_sendable=magic();
  kpost_sendable=magic();
}

#set-options "--print_implicits --split_queries always"

inline_for_extraction noextract
fn spmm
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : szp)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (#fA : perm)
  (gB : M.gpu_matrix et lB{M.is_global_matrix gB})
  (#fB : perm)
  (gC : M.gpu_matrix et lC{M.is_global_matrix gC})
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
  (#eA : ematrix et rows shared)
  // matrices densas
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  (#blockItemsK : szp)
  //(#_ : size_req rows shared cols)
  norewrite
  preserves
    cpu **
    //on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (smatrix_pts_to' gA #fA elems col_ind row_off eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    on gpu_loc (live gC) **
    pure (
      size_req rows shared cols /\
      cols /? SZ.v blockItemsK
    )
  ensures on gpu_loc (gC |-> MS.matmul eA eB)
{
  assume pure (well_formed #rows #shared #cols #gA.nnz col_ind row_off blockItemsK);
  // que raro
  let pf_size_req : squash (size_req rows shared cols) = ();
  launch_sync (
    kdesc #et #_ #rows #shared #cols #lB #lC
      gA gB gC elems col_ind row_off eA
      #eB #fA #fB
      blockItemsK #pf_size_req #()
  );
}

let _spmm_u32
  (rows shared cols : szp {SZ.fits (rows * cols) /\ SZ.fits (shared * cols)})
  =
  spmm #u32 #_
  #rows #shared #cols
  #(row_major _ _) #(row_major _ _)
