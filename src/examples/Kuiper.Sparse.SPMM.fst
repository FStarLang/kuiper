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

let slice_live
  (#et : Type0)
  (#l : nat)
  (a : gpu_array et l)
  (#[FStar.Tactics.exact (`1.0R)] f : perm)
  (i j : nat)
  : slprop
  = exists* s. gpu_pts_to_slice a #f i j s

let array_live_cell
  (#et : Type0)
  (#l : nat)
  (a :gpu_array et l)
  (#[FStar.Tactics.exact (`1.0R)] f : perm)
  (i : natlt l)
  : slprop
  = exists* v. gpu_pts_to_cell a #f i v

let matrix_live_cell
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : mlayout rows cols)
  (gm : M.gpu_matrix et lm)
  (i : natlt rows)
  (j : natlt cols)
  : slprop
  = exists* v. M.gpu_matrix_pts_to_cell gm i j v

type parameters = {
  rows : szp;
  shared : szp;
  cols : szp;
  blockItemsK : szp;
  blockItemsX : szp;
  blockWidth : (k : szp {k /? blockItemsK /\ k /? blockItemsX})
}

(* Shadow lseq to make it erased. *)
let lseq (a:Type) (n:nat) = erased (Seq.lseq a n)


let nblocks (p : parameters) : GTot pos
= p.rows * ((p.cols + p.blockItemsX - 1) / p.blockItemsX)

// TODO cambiar por blockWidth
let nthreads (p : parameters) : GTot pos
= nblocks p * p.blockWidth

inline_for_extraction noextract
let size_req (p : parameters) =
    nblocks p <= max_blocks /\
    p.blockWidth <= max_threads

let sz_nblocks (p : parameters{size_req p}) : szle max_blocks
=  p.rows *^ (p.cols /^ p.blockItemsX)

let sz_nthreads (p : parameters{size_req p}) : (nt : szp {nt <= max_threads})
= p.blockWidth

let brow (p : parameters) (bid : natlt (nblocks p))
  : GTot (natlt p.rows)
  //= bid / ((p.cols + p.blockItemsX - 1) / p.blockItemsX)
  = bid / ((p.cols + p.blockItemsX - 1) / p.blockItemsX)

let brow_ (p : parameters) (bid : szlt (nblocks p))
: Pure sz
  (requires fits (p.cols + p.blockItemsX))
  (ensures fun m -> SZ.v m == brow p bid)
  //= bid / ((p.cols + p.blockItemsX - 1) / p.blockItemsX)
  =
  assume fits (p.cols + p.blockItemsX);
  bid /^ ((p.cols +^ p.blockItemsX -^ 1sz) /^ p.blockItemsX)

let bcol (p : parameters) (bid : natlt (nblocks p))
  : GTot (natlt p.cols)
  = (bid % ((p.cols + p.blockItemsX - 1) / p.blockItemsX)) * p.blockItemsX

let bcol_ (p : parameters) (bid : szlt (nblocks p))
: Pure sz
  (requires fits (p.cols + p.blockItemsX))
  (ensures fun n -> SZ.v n == bcol p bid)
=
  (bid %^ ((p.cols +^ p.blockItemsX -^ 1sz) /^ p.blockItemsX)) *^ p.blockItemsX

// MAYBE definir threadItemsX?

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
  // necesitamos exactamente esto en una prueba
  // probablemente se puede escrbir de otra manera
  // o asumir que cada parametro del algoritmo esta acotado
  // esta bien ponerlo aca?
  fits (p.blockWidth + p.blockItemsK) /\
  // esta es rara
  fits (p.cols + p.blockItemsX)

noextract
let block_lemma whole block k
  : Lemma (requires block /? whole /\ k * block < whole)
          (ensures k * block + block <= whole)
  = ()

noextract
let block_lemma_off whole block k off
  : Lemma (requires block /? whole /\ k * block < whole /\ off < block)
          (ensures k * block + off < whole)
  = ()

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
  : slprop
  = 
  let off = ri + idx * p.blockItemsK in 
  exists* (x : et) (c : sz).
    gpu_pts_to_cell elems_tile (k * p.blockWidth + tid) x **
    gpu_pts_to_cell col_ind_tile (k * p.blockWidth + tid) c **
    pure (
      off + k * p.blockWidth + tid < re ==>
        x == elems   @! off + k * p.blockWidth + tid /\
        c == col_ind @! off + k * p.blockWidth + tid
    )

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
    if off > re then emp else
    if even it then
      (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
      (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
    else
      forall+ (k : natlt(p.blockItemsK /^ p.blockWidth)).
        barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re (it / 2) tid k

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
  : slprop
  = 
  array_live_cell elems_tile (k * p.blockWidth + tid) **
  array_live_cell col_ind_tile (k * p.blockWidth + tid)

noextract
let barrier_q_odd
  (#et : Type0)
  (p : parameters)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (elems_tile : gpu_array et p.blockItemsK)
  (col_ind_tile : gpu_array sz p.blockItemsK)
  (ri re : nat{ri <= re /\ re <= nnz})
  (idx : nat)
  (k : natlt p.blockItemsK)
  : slprop
  = 
  let off = ri + idx * p.blockItemsK in 
  exists* (x : et) (c : sz).
    gpu_pts_to_cell elems_tile #(1.0R /. p.blockWidth) k x **
    gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k c **
    pure (
      off + k < re ==>
        x == elems   @! off + k /\
        c == col_ind @! off + k
    )

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
    if off > re then emp else
      if even it then
        forall+ (k : natlt(p.blockItemsK /^ p.blockWidth)).
          barrier_q_even p nnz elems_tile col_ind_tile ri re (it / 2) tid k
      else if off + p.blockItemsK <= re
        then
          elems_tile |-> Frac (1.0R /. p.blockWidth)
            (Seq.slice elems off (off + p.blockItemsK)) **
          col_ind_tile |-> Frac (1.0R /. p.blockWidth)
            (Seq.slice col_ind off (off + p.blockItemsK))
        else
          forall+ (k : natlt p.blockItemsK).
            barrier_q_odd p elems col_ind elems_tile col_ind_tile 
              ri re (it / 2) k


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
  forall+ (k : natlt(p.blockItemsX /^ p.blockWidth)).
    when__
      (bcol p bid + k * p.blockWidth + tid < p.cols)
      (fun _ -> matrix_live_cell gC (brow p bid) (bcol p bid + k * p.blockWidth + tid))
  

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
  forall+ (k : natlt(p.blockItemsX /^ p.blockWidth)).
    when__
      (bcol p bid + k * p.blockWidth + tid < p.cols)
      (fun _ -> M.gpu_matrix_pts_to_cell gC
        (brow p bid) (bcol p bid + k * p.blockWidth + tid)
        (MS.matmul_single eA eB
          (brow p bid) (bcol p bid + k * p.blockWidth + tid)))

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
  //let (elems_tile, (col_ind_tile, _)) = sh in
  block_pre p gA gB gC elems col_ind row_off eA eB fA fB bid tid **
  (exists* (s : seq et). fst sh |-> Frac (1.0R /. p.blockWidth) s) **
  (exists* (s : seq sz). fst (snd sh) |-> Frac (1.0R /. p.blockWidth) s)

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
  //let (elems_tile, (col_ind_tile, _)) = sh in
  block_post p gA gB gC elems col_ind row_off eA eB fA fB bid tid **
  live (fst sh) #(1.0R /. p.blockWidth) **
  live (fst (snd sh)) #(1.0R /. p.blockWidth)

// TODO tal vez usar esta definicion desde arriba
let divup (n : nat) (d : pos) = ((n + d - 1) / d)

let divup_factor (n : nat) (d : pos)
= (i : natlt (divup n d) & (j : natlt d {i * d + j < n }))

let bij_divup_factor (n : nat) (d : pos)
: Kuiper.Bijection.bijection (natlt n) (divup_factor n d)
=
{
  ff = (fun (i : natlt n) -> (|i / d, i % d|) <: divup_factor n d);
  gg = (fun (|j, k|) -> j * d + k);

  ff_gg = (fun _ -> ());
  gg_ff = (fun _ -> ());
}

ghost
fn forevery_factor_
  (n : nat)
  (d : pos)
  (p : natlt n -> slprop)
  requires forall+ (i:natlt n). p i
  ensures forall+ (i1:natlt (divup n d)) (i2:natlt d {i1 * d + i2 < n}).
    p (i1 * d + i2)
{
  forevery_iso (bij_divup_factor n d) p;
  forevery_ext #(divup_factor n d)
    (fun q -> p ((bij_divup_factor n d).gg q))
    (fun q -> p (q._1 * d + q._2));
  forevery_unflatten_dep
    #(natlt (divup n d)) #(fun i1 -> (i2 : natlt d {i1 * d + i2 < n}))
    (fun i1 i2 -> p (i1 * d + i2));
}

ghost
fn forevery_unfactor_
  (n : nat)
  (d : pos)
  (p : natlt n -> slprop)
  requires forall+ (i1:natlt (divup n d)) (i2:natlt d {i1 * d + i2 < n}).
    p (i1 * d + i2)
  ensures forall+ (i:natlt n). p i
{
  forevery_flatten_dep
    #(natlt (divup n d)) #(fun i1 -> (i2 : natlt d {i1 * d + i2 < n}))
    (fun i1 i2 -> p (i1 * d + i2));
  forevery_iso (Kuiper.Bijection.bij_sym (bij_divup_factor n d )) _;
  forevery_ext _ (fun i -> p i);
}

ghost
fn forevery_ext_3
  (#a #b #c : Type0)
  (f g : a -> b -> c -> slprop)
  requires
    pure (forall x y z. f x y z == g x y z)
  requires
    forall+ (x:a) (y:b) (z:c). f x y z
  ensures
    forall+ (x:a) (y:b) (z:c). g x y z
{
  forevery_map_2
    (fun x y -> forall+ z. f x y z) 
    (fun x y -> forall+ z. g x y z)
    fn x y {
      forevery_ext (fun z -> f x y z) (fun z -> g x y z)
    };
}

ghost
fn forevery_assoc_2
  (#a:Type0)
  (#b:Type0)
  (p1 p2 p3 : a -> b -> slprop)
  requires
    forall+ (x:a) (y:b). (p1 x y ** p2 x y) ** p3 x y 
  ensures
    forall+ (x:a) (y:b). p1 x y ** p2 x y ** p3 x y
{
  forevery_map_2
    (fun x y -> (p1 x y ** p2 x y) ** p3 x y)
    (fun x y -> p1 x y ** p2 x y ** p3 x y)
    fn x y {};
}

let lem_div1 (n : nat) (d : pos) (r : natlt d)
: Lemma (requires true) (ensures (n * d + r) / d == n)
= ()

let lem_div2 (n : nat) (d : pos) (r : natlt d)
: Lemma (requires true) (ensures (n * d + r) % d == r)
= ()

ghost
fn forevery_refine_pred'
  (#a:Type0)
  (f: a -> prop)
  (p: (x:a) -> squash (f x) -> slprop)
  requires
    forall+ (x:a). when__ (f x) (p x)
  ensures
    forall+ (x:a { f x }). p x ()
{
  forevery_refine_split (fun x -> when__ (f x) (p x)) f;
  drop_ (forall+ (x:a { ~(f x) }). when__ (f x) (p x));
  forevery_ext (fun (x:a { f x }) -> when__ (f x) (p x)) (fun x -> p x ());
}


ghost
fn setup
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (#lB : mlayout p.shared p.cols)
  (#lC : mlayout p.rows p.cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#eA : ematrix et p.rows p.shared)
  // matrices densas
  (#eB : ematrix et p.shared p.cols)
  (#fA #fB : perm)
  // TODO por que toma unit?
  ()
  norewrite
  requires
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    gB |-> Frac fB eB **
    live gC
  ensures
    (forall+ (bid : natlt (nblocks p)) (tid : natlt p.blockWidth).
      block_pre p gA gB gC elems col_ind row_off eA eB fA fB bid tid) **
      emp
{
  with eC. assert gC |-> eC;
  M.gpu_matrix_explode gC;

  forevery_map
    (fun r -> 
      forall+ (c : natlt p.cols).
        M.gpu_matrix_pts_to_cell gC r c (macc eC r c)
    )
    (fun r -> 
      forall+
        (b : natlt (divup p.cols p.blockItemsX)) (tid : natlt p.blockWidth)
        (k : natlt(p.blockItemsX /^ p.blockWidth)).
          when__ (b * p.blockItemsX + k * p.blockWidth + tid < p.cols)
            (fun _ ->
              matrix_live_cell gC r
              (b * p.blockItemsX + k * p.blockWidth + tid)
            )
    )
    fn (r : natlt p.rows) {
      forevery_factor_ p.cols p.blockItemsX _;

      forevery_map
        #(natlt (divup p.cols p.blockItemsX))
        (fun b ->
          forall+ (ix : natlt p.blockItemsX { b * p.blockItemsX + ix < p.cols }).
            M.gpu_matrix_pts_to_cell gC r (b * p.blockItemsX + ix)
              (macc eC r (b * p.blockItemsX + ix))
        )
        (fun b ->
          forall+
            (tid : natlt p.blockWidth)
            (k : natlt(p.blockItemsX /^ p.blockWidth)).
            when__ (b * p.blockItemsX + k * p.blockWidth + tid < p.cols)
              (fun _ ->
                matrix_live_cell gC r
                (b * p.blockItemsX + k * p.blockWidth + tid)
              )
        )
        fn b {
          forevery_map
            #(ix : natlt p.blockItemsX { b * p.blockItemsX + ix  < p.cols })
            (fun ix -> 
              M.gpu_matrix_pts_to_cell gC r (b * p.blockItemsX + ix)
                (macc eC r (b * p.blockItemsX + ix))
            )
            (fun ix ->
              matrix_live_cell gC r (b * p.blockItemsX + ix)
            )
            fn ix {
              fold matrix_live_cell gC r (b * p.blockItemsX + ix);
            };
          forevery_unrefine_pred' #(natlt p.blockItemsX)
            (fun ix -> b * p.blockItemsX + ix  < p.cols)
            (fun ix _ -> matrix_live_cell gC r (b * p.blockItemsX + ix));

          forevery_factor p.blockItemsX
            (p.blockItemsX /^ p.blockWidth) p.blockWidth
            (fun ix ->
              when__ (b * p.blockItemsX + ix < p.cols)
                (fun _ -> matrix_live_cell gC r (b * p.blockItemsX + ix))
            );
          
          forevery_commute
            #(natlt (p.blockItemsX /^ p.blockWidth)) #(natlt p.blockWidth)
            (fun k tid ->
              when__ (b * p.blockItemsX + (k * p.blockWidth + tid) < p.cols)
                (fun _ -> matrix_live_cell gC r (b * p.blockItemsX + (k * p.blockWidth + tid)))
            );
          forevery_ext_2
            #(natlt p.blockWidth)
            #(natlt (p.blockItemsX /^ p.blockWidth))
            (fun tid k ->
              when__ (b * p.blockItemsX + (k * p.blockWidth + tid) < p.cols)
                (fun _ -> matrix_live_cell gC r (b * p.blockItemsX + (k * p.blockWidth + tid)))
            )
            (fun tid k ->
              when__ (b * p.blockItemsX + k * p.blockWidth + tid < p.cols)
                (fun _ -> matrix_live_cell gC r (b * p.blockItemsX + k * p.blockWidth + tid))
            );
        };
    }; 

  forevery_unfactor' (nblocks p) _ _ _;
  forevery_ext_3
    #(natlt (nblocks p))
    #(natlt p.blockWidth)
    #(natlt (p.blockItemsX /^ p.blockWidth))
    _
    (fun bid tid k ->
      when__ (bcol p bid + k * p.blockWidth + tid < p.cols)
        (fun _ ->
          matrix_live_cell gC
            (brow p bid)
            (bcol p bid + k * p.blockWidth + tid)
        )
    );

  smatrix_share_n' gA #fA elems col_ind row_off eA (nthreads p);//elems col_ind row_off eA; 

  M.gpu_matrix_share_n gB (nthreads p) #fB;

  forevery_zip #(natlt (nthreads p))
   _
   (fun _ -> M.gpu_matrix_pts_to gB #(fB /. (nthreads p)) eB);

  forevery_factor (nthreads p) (nblocks p) p.blockWidth _;

  forevery_zip_2
    (fun _ _ ->
      smatrix_pts_to' gA #(fA /. (nthreads p)) elems col_ind row_off eA **
      M.gpu_matrix_pts_to gB #(fB /. (nthreads p)) eB
    )
    _;

  forevery_assoc_2 _ _ _;

  ();
}

ghost
fn block_setup
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (#lB : mlayout p.shared p.cols)
  (#lC : mlayout p.rows p.cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#eA : ematrix et p.rows p.shared)
  // matrices densas
  (#eB : ematrix et p.shared p.cols)
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc et p ))
  (bid : natlt (nblocks p))
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt p.blockWidth).
      block_pre p gA gB gC elems col_ind row_off eA eB fA fB bid tid)
  ensures
    (forall+ (tid : natlt p.blockWidth).
      kpre p gA gB gC elems col_ind row_off eA eB fA fB sh bid tid) **
      emp
{
  unfold_c_shmems sh (`%shmems_desc);
  with (x : seq _). assert fst sh |-> x;
  with (c : seq _). assert fst (snd sh) |-> c;

  gpu_slice_share (fst sh) 0 p.blockItemsK p.blockWidth;
  forevery_map #(natlt p.blockWidth)
    (fun _ -> fst sh |-> Frac (1.0R /. p.blockWidth) x)
    (fun _ -> (exists* (s : seq _). fst sh |-> Frac (1.0R /. p.blockWidth) s))
    fn _ {}; 
  
  gpu_slice_share (fst (snd sh)) 0 p.blockItemsK p.blockWidth;
  forevery_map #(natlt p.blockWidth)
    (fun _ -> fst (snd sh) |-> Frac (1.0R /. p.blockWidth) c)
    (fun _ -> (exists* (s : seq _). fst (snd sh) |-> Frac (1.0R /. p.blockWidth) s))
    fn _ {}; 

  forevery_zip3 #(natlt p.blockWidth)
    _
    (fun _ -> (exists* (s : seq _). fst sh |-> Frac (1.0R /. p.blockWidth) s))
    (fun _ -> (exists* (s : seq _). fst (snd sh) |-> Frac (1.0R /. p.blockWidth) s));
  
}

ghost
fn block_teardown
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (#lB : mlayout p.shared p.cols)
  (#lC : mlayout p.rows p.cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#eA : ematrix et p.rows p.shared)
  // matrices densas
  (#eB : ematrix et p.shared p.cols)
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc et p ))
  (bid : natlt (nblocks p))
  ()
  norewrite
  requires
    (forall+ (tid : natlt p.blockWidth).
      kpost p gA gB gC elems col_ind row_off eA eB fA fB sh bid tid) **
      emp
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt p.blockWidth).
      block_post p gA gB gC elems col_ind row_off eA eB fA fB bid tid)
{
  forevery_unzip3 _ _ _;
  forevery_natlt_pop p.blockWidth
    (fun tid -> exists* x.
      gpu_pts_to_slice (fst sh) #(1.0R /. p.blockWidth) 0 p.blockItemsK x
    );
  with elems_tile.
    assert gpu_pts_to_slice (fst sh) #(1.0R /. p.blockWidth) 0 p.blockItemsK
      elems_tile;
  forevery_map_extra #(natlt (p.blockWidth - 1))
    (gpu_pts_to_slice (fst sh) #(1.0R /. p.blockWidth) 0 p.blockItemsK
      elems_tile)
    (fun tid -> exists* x.
      gpu_pts_to_slice (fst sh) #(1.0R /. p.blockWidth) 0 p.blockItemsK x
    )
    (fun tid ->
      gpu_pts_to_slice (fst sh) #(1.0R /. p.blockWidth) 0 p.blockItemsK
        elems_tile
    )
    fn tid {
      gpu_slice_pts_to_eq (fst sh) 0 p.blockItemsK (1.0R /. p.blockWidth)
        #_ #elems_tile;
    };
  forevery_natlt_push p.blockWidth
    (fun tid ->
      gpu_pts_to_slice (fst sh) #(1.0R /. p.blockWidth) 0 p.blockItemsK
        elems_tile
    );

  forevery_natlt_pop p.blockWidth
    (fun tid -> exists* x.
      gpu_pts_to_slice (fst (snd sh)) #(1.0R /. p.blockWidth) 0 p.blockItemsK x
    );
  with col_ind_tile.
    assert gpu_pts_to_slice (fst (snd sh)) #(1.0R /. p.blockWidth) 0 p.blockItemsK
      col_ind_tile;
  forevery_map_extra #(natlt (p.blockWidth - 1))
    (gpu_pts_to_slice (fst (snd sh)) #(1.0R /. p.blockWidth) 0 p.blockItemsK
      col_ind_tile)
    (fun tid -> exists* x.
      gpu_pts_to_slice (fst (snd sh)) #(1.0R /. p.blockWidth) 0 p.blockItemsK x
    )
    (fun tid ->
      gpu_pts_to_slice (fst (snd sh)) #(1.0R /. p.blockWidth) 0 p.blockItemsK
        col_ind_tile
    )
    fn tid {
      gpu_slice_pts_to_eq (fst (snd sh)) 0 p.blockItemsK (1.0R /. p.blockWidth)
        #_ #col_ind_tile;
    };
  forevery_natlt_push p.blockWidth
    (fun tid ->
      gpu_pts_to_slice (fst (snd sh)) #(1.0R /. p.blockWidth) 0 p.blockItemsK
        col_ind_tile
    );

  gpu_slice_gather (fst sh) 0 p.blockItemsK p.blockWidth;
  gpu_slice_gather (fst (snd sh)) 0 p.blockItemsK p.blockWidth;

  fold_c_shmems sh (`%shmems_desc);

  ();
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (#lB : mlayout p.shared p.cols)
  (#lC : mlayout p.rows p.cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#eA : ematrix et p.rows p.shared)
  // matrices densas
  (#eB : ematrix et p.shared p.cols)
  (#fA #fB : perm)
  // TODO por que toma unit?
  ()
  norewrite
  requires
    // TODO falta el frame aca??
    (forall+ (bid : natlt (nblocks p)) (tid : natlt p.blockWidth).
      block_post p gA gB gC elems col_ind row_off eA eB fA fB bid tid) **
      emp
  ensures
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    gB |-> Frac fB eB **
    gC |-> MS.matmul eA eB
{
  forevery_unzip_2 _ _;
  forevery_unfactor' (nthreads p) _ _
    (fun _ _ ->
      smatrix_pts_to' gA #(fA /. nthreads p) elems col_ind row_off eA
    );
  smatrix_gather_n' gA #fA elems col_ind row_off eA (nthreads p);//elems col_ind row_off eA; 

  forevery_unzip_2 _ _;
  forevery_unfactor' (nthreads p) _ _
    (fun _ _ -> gB |-> Frac (fB /. nthreads p) eB);
  M.gpu_matrix_gather_n gB (nthreads p) #fB;


  forevery_map
    (fun bid -> 
      forall+
        (tid: natlt (v p.blockWidth))
        (k: natlt (v (SizeT.div p.blockItemsX p.blockWidth))).
        when__ (bcol p bid + k * v p.blockWidth + tid < v p.cols)
          (fun _ ->
            M.gpu_matrix_pts_to_cell gC
              (brow p bid)
              (bcol p bid + k * v p.blockWidth + tid)
              (MS.matmul_single eA
                  eB
                  (brow p bid)
                  (bcol p bid + k * v p.blockWidth + tid)))
    ) 
    (fun bid ->
      forall+ (ix : natlt p.blockItemsX).
        when__
          (bcol p bid + ix < p.cols)
          (fun _ ->
            M.gpu_matrix_pts_to_cell gC
              (brow p bid)
              (bcol p bid + ix)
              (MS.matmul_single eA eB
                (brow p bid)
                (bcol p bid + ix)
              )
          )
    )
    fn bid {
      forevery_ext_2
        #(natlt p.blockWidth)
        #(natlt (p.blockItemsX /^ p.blockWidth))
        (fun tid k -> 
          when__ (bcol p bid + k * v p.blockWidth + tid < v p.cols)
            (fun _ ->
              M.gpu_matrix_pts_to_cell gC
                (brow p bid)
                (bcol p bid + k * v p.blockWidth + tid)
                (MS.matmul_single eA
                    eB
                    (brow p bid)
                    (bcol p bid + k * v p.blockWidth + tid)))
        )
        (fun tid k ->
          when__ (bcol p bid + (k * v p.blockWidth + tid) < v p.cols)
            (fun _ ->
              M.gpu_matrix_pts_to_cell gC
                (brow p bid)
                (bcol p bid + (k * v p.blockWidth + tid))
                (MS.matmul_single eA
                    eB
                    (brow p bid)
                    (bcol p bid + (k * v p.blockWidth + tid)))
            )
        );
      forevery_commute _;
      forevery_unfactor p.blockItemsX (p.blockItemsX /^ p.blockWidth) _
        (fun ix ->
          when__
            (bcol p bid + ix < p.cols)
            (fun _ ->
              M.gpu_matrix_pts_to_cell gC
                (brow p bid)
                (bcol p bid + ix)
                (MS.matmul_single eA eB
                  (brow p bid)
                  (bcol p bid + ix)
                )
            )
        );
    };
  forevery_factor (nblocks p) p.rows (divup p.cols p.blockItemsX) _;
  forevery_map #(natlt p.rows)
    (fun r ->
      forall+
        (b: natlt (divup p.cols p.blockItemsX))
        (ix : natlt p.blockItemsX).
        when__
          (bcol p (r * divup p.cols p.blockItemsX + b) + ix < p.cols)
          (fun _ ->
            M.gpu_matrix_pts_to_cell gC
              (brow p (r * divup p.cols p.blockItemsX + b))
              (bcol p (r * divup p.cols p.blockItemsX + b) + ix)
              (MS.matmul_single eA eB
                (brow p (r * divup p.cols p.blockItemsX + b))
                (bcol p (r * divup p.cols p.blockItemsX + b) + ix)
              )
          )
    )
    (fun r ->
      forall+ (c : natlt p.cols).
        M.gpu_matrix_pts_to_cell gC r c (MS.matmul_single eA eB r c)
    )
    fn r {
      forevery_map_2 #(natlt (divup p.cols p.blockItemsX)) #(natlt p.blockItemsX)
        (fun b ix ->
          when__
            (bcol p (r * divup p.cols p.blockItemsX + b) + ix < p.cols)
            (fun _ ->
              M.gpu_matrix_pts_to_cell gC
                (brow p (r * divup p.cols p.blockItemsX + b))
                (bcol p (r * divup p.cols p.blockItemsX + b) + ix)
                (MS.matmul_single eA eB
                  (brow p (r * divup p.cols p.blockItemsX + b))
                  (bcol p (r * divup p.cols p.blockItemsX + b) + ix)
                )
            )
        )
        (fun b ix ->
          when__
            (b * p.blockItemsX + ix < p.cols)
            (fun _ ->
              M.gpu_matrix_pts_to_cell gC r (b * p.blockItemsX + ix)
                (MS.matmul_single eA eB r (b * p.blockItemsX + ix))
            )
        )
        fn b ix {
          lem_div1 r (divup p.cols p.blockItemsX) b;
          rewrite each (brow p (r * divup p.cols p.blockItemsX + b))
            as r;
          lem_div2 r (divup p.cols p.blockItemsX) b;
          rewrite each (bcol p (r * divup p.cols p.blockItemsX + b))
            as (b * p.blockItemsX);
        };
      forevery_map #(natlt (divup p.cols p.blockItemsX))
        (fun b ->
          forall+ (ix : natlt p.blockItemsX).
            when__
              (b * p.blockItemsX + ix < p.cols)
              (fun _ ->
                M.gpu_matrix_pts_to_cell gC r (b * p.blockItemsX + ix)
                  (MS.matmul_single eA eB r (b * p.blockItemsX + ix))
              )
        )
        (fun b ->
          forall+ (ix : natlt p.blockItemsX {b * p.blockItemsX + ix < p.cols}).
            M.gpu_matrix_pts_to_cell gC r (b * p.blockItemsX + ix)
              (MS.matmul_single eA eB r (b * p.blockItemsX + ix))
        )
        fn b {
          forevery_refine_pred' #(natlt p.blockItemsX)
            (fun ix -> b * p.blockItemsX + ix < p.cols) _;
        };

      forevery_unfactor_
        p.cols
        (p.blockItemsX)
        (fun c ->
          M.gpu_matrix_pts_to_cell gC r c
            (MS.matmul_single eA eB r c)
        );
    };

  forevery_ext_2
    (fun r c -> M.gpu_matrix_pts_to_cell gC r c (MS.matmul_single eA eB r c))
    (fun r c -> M.gpu_matrix_pts_to_cell gC r c (macc (MS.matmul eA eB) r c));
  M.gpu_matrix_implode gC;

  ();
}

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
  (ri : sz{ri == row_off @! brow p bid})
  (re : sz{re == row_off @! brow p bid + 1})
  (idx : nat)
  (tid : natlt p.blockWidth)
  requires
    pure (ri + idx * p.blockItemsK <= re) **
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
  (ri : sz{ri == row_off @! brow p bid})
  (re : sz{re == row_off @! brow p bid + 1})
  (idx : nat)
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + idx * p.blockItemsK <= re))
  requires
    forall+ (k: natlt (p.blockItemsK /^ p.blockWidth)).
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
  ensures
    barrier_p p elems col_ind row_off elems_tile col_ind_tile bid (idx * 2 + 1) tid
{
  assert rewrites_to ri (row_off @! brow p bid);
  assert rewrites_to re (row_off @! brow p bid + 1);

  let it = idx * 2 + 1;

  rewrite each idx as (it / 2);
  rewrite forall+ (k: natlt (p.blockItemsK /^ p.blockWidth)).
    barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re (it / 2) tid k
    as (
      let off = ri + (it / 2) * p.blockItemsK in
      if off > re then emp else
      if even it then
        (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
        (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
      else
        forall+ (k : natlt(p.blockItemsK /^ p.blockWidth)).
          barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re (it / 2) tid k
    );

  fold barrier_p p elems col_ind row_off elems_tile col_ind_tile bid it tid;

  rewrite each it as (idx * 2 + 1);

  ();
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
  (ri : sz{ri == row_off @! brow p bid})
  (re : sz{re == row_off @! brow p bid + 1})
  (idx : nat)
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + idx * p.blockItemsK <= re))
  requires
    barrier_q p elems col_ind row_off elems_tile col_ind_tile bid (idx * 2) tid
  ensures
    forall+ (k : natlt (p.blockItemsK /^ p.blockWidth)).
      barrier_q_even p nnz elems_tile col_ind_tile ri re idx tid k
{
  assert rewrites_to ri (row_off @! brow p bid);
  assert rewrites_to re (row_off @! brow p bid + 1);

  let it = idx * 2;

  rewrite each (idx * 2) as it;
  unfold barrier_q p elems col_ind row_off elems_tile col_ind_tile bid it tid;


  rewrite each (ri + (it / 2) * p.blockItemsK > re) as false;
  rewrite each (even it) as true;

  rewrite each it as (idx * 2);
  rewrite each (idx * 2 / 2) as idx;

  ();

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
  (ri : sz{ri == row_off @! brow p bid})
  (re : sz{re == row_off @! brow p bid + 1})
  (idx : nat)
  (tid : natlt p.blockWidth)
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
  assert rewrites_to ri (row_off @! brow p bid);
  assert rewrites_to re (row_off @! brow p bid + 1);

  let it = idx * 2 + 1;

  rewrite each (idx * 2 + 1) as it;
  unfold barrier_q p elems col_ind row_off elems_tile col_ind_tile bid it tid;

  rewrite each (ri + (it / 2) * p.blockItemsK > re) as false;
  rewrite each (even it) as false;
  rewrite each (ri + (it / 2) * p.blockItemsK + p.blockItemsK <= re) as true;

  rewrite each it as (idx * 2 + 1);
  rewrite each ((idx * 2 + 1) / 2) as idx;

  ();
}

ghost
fn barrier_q_unfold_odd_residue
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
  (ri : sz{ri == row_off @! brow p bid})
  (re : sz{re == row_off @! brow p bid + 1})
  (idx : nat)
  (tid : natlt p.blockWidth)
  (#_ : squash (ri + idx * p.blockItemsK <= re))
  (#_ : squash (ri + idx * p.blockItemsK + p.blockItemsK > re))
  requires
    barrier_q p elems col_ind row_off elems_tile col_ind_tile
      bid (idx * 2 + 1) tid
  ensures forall+ (k : natlt p.blockItemsK).
    barrier_q_odd p elems col_ind elems_tile col_ind_tile ri re idx k
{
  assert rewrites_to ri (row_off @! brow p bid);
  assert rewrites_to re (row_off @! brow p bid + 1);

  let it = idx * 2 + 1;

  rewrite each (idx * 2 + 1) as it;
  unfold barrier_q p elems col_ind row_off elems_tile col_ind_tile bid it tid;

  rewrite each (ri + (it / 2) * p.blockItemsK > re) as false;
  rewrite each (even it) as false;
  rewrite each (ri + (it / 2) * p.blockItemsK + p.blockItemsK <= re) as false;

  rewrite each it as (idx * 2 + 1);
  rewrite each ((idx * 2 + 1) / 2) as idx;

  ();
}

open Kuiper.Bijection

let natlt_refined_bij (m n : nat)
: bijection (a : natlt m {a < n}) (natlt (min m n))
= {
  ff = (fun (a : natlt m {a < n}) -> let a' : natlt (min m n) = a in a');
  gg = (fun (b : natlt (min m n)) -> b);
  ff_gg = (fun b -> ());
  gg_ff = (fun a -> ())
}

let natlt_is_between (n : nat) : Lemma (natlt n == between 0 n)
  =
  FStar.RefinementExtensionality.refext
    nat
    (fun (x:nat) -> x < n)
    (fun (x:nat) -> 0 <= x /\ x < n);
  assert (x:nat{x < n} == x:nat{0 <= x /\ x < n});
  assert (natlt n == x:nat{x < n});
  assert_norm (between 0 n == x:nat{0 <= x /\ x < n});
  ()


inline_for_extraction noextract
fn foreach
  (n : sz)
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
  natlt_is_between n;
  assert pure (natlt n == between 0sz n);
  forevery_rw_type (natlt n) (between 0sz n) p;
  Kuiper.For.for_loop' 0sz n
    p q
    frame
    fn x { f x };
  forevery_rw_type (between 0sz n) (natlt n) q;
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
  (ri re : sz{ri < re /\ re <= gA.nnz})
  (idx : sz)
  (tid : szlt (p.blockWidth))
  (k : szlt (p.blockItemsK /^ p.blockWidth))
  (#_ : squash (ri + idx * p.blockItemsK + k * p.blockWidth + tid < re))
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
  unfold array_live_cell elems_tile;
  unfold array_live_cell col_ind_tile;

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
  
  fold barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k;
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
  (ri : sz{ri == row_off @! brow p bid})
  (re : sz{re == row_off @! brow p bid + 1})
  (idx : sz)
  (tid : szlt p.blockWidth)
  (#_ : squash(ri + idx * p.blockItemsK + p.blockItemsK <= re))
  norewrite
  preserves
    gpu ** 
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    B.barrier_tok (barrier_contract p elems col_ind row_off elems_tile col_ind_tile bid) **
    thread_id p.blockWidth tid
  requires
    B.barrier_state (idx * 2) **
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
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
    bid ri re idx tid;

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
    bid ri re idx tid;
        
  foreach (p.blockItemsK /^ p.blockWidth)
    (fun ki -> barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid ki)
    (fun ki -> barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid ki) 
    (fun k ->
      sparse_load_one p gA #row_off #elems #col_ind #eA elems_tile col_ind_tile
        bid ri re idx tid k
    );

  barrier_p_fold_odd p elems col_ind row_off elems_tile col_ind_tile
    bid ri re idx tid;

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
    bid ri re idx tid;


  let elems_slice   : erased (seq et) = Seq.slice elems   off (off + p.blockItemsK);
  let col_ind_slice : erased (seq sz) = Seq.slice col_ind off (off + p.blockItemsK);

  // TODO: mover esto a la prueba de la barrera
  //forevery_map
    //(barrier_q_odd p elems col_ind elems_tile col_ind_tile
      //ri re idx)
    //(fun k ->
      //gpu_pts_to_cell elems_tile #(1.0R /. p.blockWidth) k
        //(elems_slice @! k) **
      //gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k
        //(col_ind_slice @! k)
    //)
    //fn k
    //{
      //unfold barrier_q_odd;
      //()
    //};
  //forevery_unzip #(natlt p.blockItemsK)
    //(fun k -> gpu_pts_to_cell elems_tile #(1.0R /. p.blockWidth) k
        //(elems_slice @! k))
    //(fun k -> gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k
        //(col_ind_slice @! k));


  //gpu_array_unslice_1 elems_tile #(1.0R /. p.blockWidth);
  //gpu_array_unslice_1 col_ind_tile #(1.0R /. p.blockWidth) #col_ind_slice;

  ();
}

let between_coerce_down
  (#i #j #j' : nat{i < j' /\ j' <= j})
  (k : between i j{k < j'})
: between i j'
= k

unfold
let between_coerce_up
  (#i #j #i' : nat{i <= i' /\ i' < j})
  (k : between i j{i' <= k})
: between i' j
= k

let between_restrict_shift_down (i j j' : nat { i < j' /\ j' <= j }) (p: between i j' -> slprop) =
  forevery_refine_ext' #nat #(fun k -> i <= k /\ k < j /\ k < j')
    (fun k -> i <= k /\ k < j') (fun k -> p k)

let between_restrict_shift_up (i j i' : nat { i <= i' /\ i' < j }) (p: between i' j -> slprop) =
  forevery_refine_ext' #nat #(fun k -> i <= k /\ k < j /\ i' <= k)
    (fun k -> i' <= k /\ k < j) (fun k -> p k)

ghost
fn forevery_between_restrict_down
  (i j j' : nat{i < j' /\ j' <= j})
  (p : between i j' -> slprop)
  requires forall+ (k : between i j {k < j'}). p (between_coerce_down k)
  ensures  forall+ (k : between i j'). p k
{
  between_restrict_shift_down i j j' p;
} 

ghost
fn forevery_between_restrict_up
  (i j i' : nat{i <= i' /\ i' < j})
  (p : between i' j -> slprop)
  requires forall+ (k : between i j {i' <= k}). p (between_coerce_up k)
  ensures  forall+ (k : between i' j). p k
{
  between_restrict_shift_up i j i' p;
} 

ghost
fn rec gpu_forall_cell_to_slice_
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (i j : nat {i < j})
  (#v : erased (seq a))
  (#_ : squash (Seq.length v == j - i))
  requires
    (forall+ (k : between i j).
      gpu_pts_to_cell arr #f k (v @! k - i))
  ensures gpu_pts_to_slice arr #f i j v
  decreases j
{
  let j' = j - 1;
  if (j' = i) {
    forevery_singleton_elim' #(between i j) _ j';
    assert pure (Seq.equal seq![v @! 0] v);
    rewrite gpu_pts_to_slice arr #f j' (j' + 1) seq![v @! 0]
      as gpu_pts_to_slice arr #f i j v;
    ()
  } else {
    forevery_remove #(between i j)
      (fun k -> gpu_pts_to_cell arr #f k (v @! k - i))
      j';
    forevery_refine_ext #(between i j)
      (fun k -> k < j')
      (fun k -> gpu_pts_to_cell arr #f k (v @! k - i));
    forevery_between_restrict_down i j j'
      (fun k -> gpu_pts_to_cell arr #f k (v @! k - i));
    forevery_ext #(between i j')
      (fun k -> gpu_pts_to_cell arr #f k (v @! k - i))
      (fun k -> gpu_pts_to_cell arr #f k (Seq.slice v 0 (j' - i) @! k - i));
    gpu_forall_cell_to_slice_ arr i j';
    gpu_slice_concat arr #f i _ _;
    assert pure (Seq.equal (Seq.append (Seq.slice v 0 (j' - i)) seq![v @! j' - i]) v);
    rewrite gpu_pts_to_slice arr #f i (j' + 1) (Seq.append (Seq.slice v 0 (j' - i)) seq![v @! j' - i])
      as gpu_pts_to_slice arr #f i j v;
  }
}

ghost
fn gpu_forall_cell_to_slice
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (i n #m : nat {i <= n /\ n <= m})
  (#v : erased (seq a))
  (#_ : squash (Seq.length v == n - i))
  requires
    (forall+ (k : between i n).
      gpu_pts_to_cell arr #f k (v @! k - i))
  preserves
    slice_live arr #f n m 
  ensures gpu_pts_to_slice arr #f i n v
{
  if (i < n)
  {
    gpu_forall_cell_to_slice_ arr #f i n
  }
  else {
    forevery_elim_empty _;
    unfold slice_live;
    assert pure (Seq.empty `Seq.equal` v);

    with s. assert gpu_pts_to_slice arr #f n m s;
    assert pure (Seq.append v s `Seq.equal` s);
    assert gpu_pts_to_slice arr #f n m (Seq.append v s);
    gpu_slice_split arr #f #v #s n n m;


    fold slice_live arr #f n m;
    rewrite gpu_pts_to_slice arr #f n n v
    as gpu_pts_to_slice arr #f i n v;
  };

}

unfold
let coerce_fun (#a : Type0) (#b #c : Type{a == b}) (p : a -> c) (x : b) : c = p x 

ghost
fn forevery_rw_type_ref
  (a:Type0)
  (b:Type{a == b})
  (p : a -> prop)
  (f : a -> slprop)
  requires
    forall+ (x:a{p x}). f x
  ensures
    forall+ (x:b{p x}). coerce_fun #a #b f x
{
  forevery_rw_type (x : a{p x}) (x : b{p x}) f;
}

let between_to_natlt (#m #n : nat{m <= n}) (a : between m n) : natlt (n - m) = a - m
let natlt_to_between (#m #n : nat{m <= n}) (a : natlt (n - m)) : between m n = a + m

let bij_between_natlt (m n : nat{m <= n})
: bijection (between m n) (natlt (n - m))
= {
  ff = between_to_natlt;
  gg = natlt_to_between;
  ff_gg = (fun b -> ());
  gg_ff = (fun a -> ())
}

instance enumerable_between (m n:nat{m <= n}) : enumerable (between m n) = {
  _cardinal = n - m;
  bij = bij_between_natlt m n;
}

ghost
fn gpu_forall_live_cell_to_slice
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (i j : nat {i < j})
  requires forall+ (k : between i j).
    exists* x. gpu_pts_to_cell arr #f k x
  ensures exists* v.
    gpu_pts_to_slice arr #f i j v
{
  let y = forevery_exists #(between i j) (gpu_pts_to_cell arr #f);
  let v = Seq.init_ghost (j - i) (fun k -> y (k + i));
  forevery_ext #(between i j)
    (fun k -> gpu_pts_to_cell arr #f k (y k))
    (fun k -> gpu_pts_to_cell arr #f k (v @! k - i));
  gpu_forall_cell_to_slice_ arr i j;
}

inline_for_extraction noextract
fn sparse_load_residue
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
  (ri : sz{ri == row_off @! brow p bid})
  (re : sz{re == row_off @! brow p bid + 1})
  (idx : sz)
  (tid : szlt p.blockWidth)
  (#_ : squash(ri + idx * p.blockItemsK <= re))
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
    pure (
      re - (ri + idx * p.blockItemsK) < p.blockItemsK
    )
  ensures
    B.barrier_state ((idx + 1) * 2) **
    gpu_pts_to_slice elems_tile #(1.0R /. p.blockWidth)
      0 (re - (ri + idx * p.blockItemsK))
      (Seq.slice elems (ri + idx * p.blockItemsK) re) **
    gpu_pts_to_slice col_ind_tile #(1.0R /. p.blockWidth)
      0 (re - (ri + idx * p.blockItemsK))
      (Seq.slice col_ind (ri + idx * p.blockItemsK) re) **
    slice_live elems_tile #(1.0R /. p.blockWidth)
      (re - (ri + idx * p.blockItemsK)) p.blockItemsK **
    slice_live col_ind_tile #(1.0R /. p.blockWidth)
      (re - (ri + idx * p.blockItemsK)) p.blockItemsK
{

  let off = ri +^ idx *^ p.blockItemsK;

  barrier_p_fold_even p elems col_ind row_off elems_tile col_ind_tile
    bid ri re idx tid;

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
    bid ri re idx tid;

  let tresidue : sz = (re -^ off +^ (p.blockWidth -^ 1sz) -^ tid) /^ p.blockWidth;

  forevery_refine_split #(natlt (p.blockItemsK /^ p.blockWidth))
    (barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid)
    (fun k ->
      k < SZ.v tresidue);
  
  forevery_natlt_restrict #tresidue
    (p.blockItemsK /^ p.blockWidth)
    (fun (k : natlt tresidue) ->
      barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid (natlt_coerce k));
  
  foreach tresidue
    (fun (ki : natlt tresidue) ->
      barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid (natlt_coerce ki))
    (fun (ki : natlt tresidue) ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid (natlt_coerce ki))
    #(
      gpu ** 
      smatrix_pts_to' gA #fA elems col_ind row_off eA **
      B.barrier_tok (barrier_contract p elems col_ind row_off elems_tile col_ind_tile bid) **
      thread_id p.blockWidth tid
    )
    fn (k : natlt tresidue)
    {
      sparse_load_one p gA #row_off #elems #col_ind #eA elems_tile col_ind_tile
        bid ri re idx tid k;
    };

  forevery_natlt_extend (p.blockItemsK /^ p.blockWidth)
    (fun (k : natlt tresidue) -> 
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid (natlt_coerce k)
    );

  forevery_ext #(k : natlt (p.blockItemsK /^ p.blockWidth){k < tresidue})
    (fun k -> 
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid
        (natlt_coerce (natlt_coerce #tresidue k))
    )
    (fun k -> 
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
    );

  forevery_map #(k : natlt (p.blockItemsK /^ p.blockWidth) {~(k < tresidue)})
    (fun k ->
      barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid k
    )
    (fun k ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
    )
    fn k {
      unfold barrier_q_even;
      unfold array_live_cell elems_tile;
      unfold array_live_cell col_ind_tile;
      fold barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k;
    };

  forevery_refine_join 
    (fun k ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
    )
    (fun k -> k < tresidue) (fun k -> ~(k < tresidue));

  forevery_unrefine (fun k ->
    barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
  );

  barrier_p_fold_odd p elems col_ind row_off elems_tile col_ind_tile
    bid ri re idx tid;

  rewrite barrier_p p elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2 + 1) tid
       as (barrier_contract p elems col_ind row_off elems_tile col_ind_tile bid).rin
            (idx * 2 + 1) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p elems col_ind row_off elems_tile col_ind_tile bid).rout
            (idx * 2 + 1) tid
       as barrier_q p elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2 + 1) tid;

  barrier_q_unfold_odd_residue p elems col_ind row_off elems_tile col_ind_tile
    bid ri re idx tid;
  
  forevery_refine_split
    (barrier_q_odd p elems col_ind elems_tile col_ind_tile ri re idx)
    (fun (k : natlt p.blockItemsK) -> k < re - off);

  let elems_slice   : erased (seq et) = Seq.slice elems   off re;
  let col_ind_slice : erased (seq sz) = Seq.slice col_ind off re;

  // el residuo
  forevery_map #(k : natlt p.blockItemsK {k < re - off})
    (barrier_q_odd p elems col_ind elems_tile col_ind_tile
      ri re idx)
    (fun k ->
      gpu_pts_to_cell elems_tile #(1.0R /. p.blockWidth) k
        (elems_slice @! k) **
      gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k
        (col_ind_slice @! k)
    )
    fn k { unfold barrier_q_odd };
  
  forevery_natlt_restrict #(re - off) p.blockItemsK
    (fun k ->
      gpu_pts_to_cell elems_tile #(1.0R /. p.blockWidth) k
        (elems_slice @! k) **
      gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k
        (col_ind_slice @! k)
    );

  natlt_is_between (re - off);
  forevery_rw_type (natlt (re - off)) (between 0 (re - off)) _;

  forevery_ext #(between 0 (re - off))
    (fun k ->
      gpu_pts_to_cell elems_tile #(1.0R /. p.blockWidth) k
        (elems_slice @! k) **
      gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k
        (col_ind_slice @! k))
    (fun k ->
      gpu_pts_to_cell elems_tile #(1.0R /. p.blockWidth) k
        (elems_slice @! k - 0) **
      gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k
        (col_ind_slice @! k - 0));

  forevery_unzip #(between 0 (re - off))
    (fun k -> gpu_pts_to_cell elems_tile #(1.0R /. p.blockWidth) k
        (elems_slice @! k - 0))
    (fun k -> gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k
        (col_ind_slice @! k - 0));


  // el resto
  forevery_map #(k : natlt p.blockItemsK {~(k < re - off)})
    (barrier_q_odd p elems col_ind elems_tile col_ind_tile
      ri re idx)
    (fun k ->
      (exists* x. gpu_pts_to_cell elems_tile   #(1.0R /. p.blockWidth) k x) **
      (exists* c. gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k c)
    )
    fn k { unfold barrier_q_odd };


  natlt_is_between p.blockItemsK;
  forevery_rw_type_ref
    (natlt p.blockItemsK)
    (between 0 p.blockItemsK)
    (fun (k : natlt p.blockItemsK) -> ~(k < re - off))
    (fun (k : natlt p.blockItemsK) -> 
      (exists* x. gpu_pts_to_cell elems_tile   #(1.0R /. p.blockWidth) k x) **
      (exists* c. gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k c)
    );


  forevery_refine_ext #(between 0 p.blockItemsK)
    (fun k -> re - off <= k)
    (fun k -> 
      (exists* x. gpu_pts_to_cell elems_tile   #(1.0R /. p.blockWidth) k x) **
      (exists* c. gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k c)
    );

  forevery_between_restrict_up 0 p.blockItemsK (re - off)
    (fun k -> 
      (exists* x. gpu_pts_to_cell elems_tile   #(1.0R /. p.blockWidth) k x) **
      (exists* c. gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k c));

  forevery_unzip #(between (re - off) p.blockItemsK)
    (fun k -> 
      (exists* x. gpu_pts_to_cell elems_tile   #(1.0R /. p.blockWidth) k x))
    (fun k ->
      (exists* c. gpu_pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k c));



  gpu_forall_live_cell_to_slice elems_tile   (re - off) p.blockItemsK;  
  gpu_forall_live_cell_to_slice col_ind_tile (re - off) p.blockItemsK;  

  fold slice_live elems_tile   #(1.0R /. p.blockWidth) (re - off) p.blockItemsK;
  fold slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK;

  gpu_forall_cell_to_slice elems_tile 0 (re - off);  
  gpu_forall_cell_to_slice col_ind_tile 0 (re - off);
  

  rewrite each off as (ri +^ idx *^ p.blockItemsK);

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
  (n_idx : szlt p.cols {SZ.v n_idx == bcol p bid})
  norewrite
  preserves
    gpu **
    elems_tile |-> Frac (1.0R /. p.blockWidth) v_elems **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth) v_col_ind **
    gB |-> Frac (fB /. nthreads p) eB
  requires
    pure (fits (p.cols + p.blockItemsX)) **
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
      block_lemma_off p.blockItemsX p.blockWidth !x tid;

      assert pure (!x *^ p.blockWidth + tid < p.blockItemsX);
      let dense_off : sz = n_idx +^ !x *^ p.blockWidth +^ tid;

      if (dense_off <^ p.cols)
        ensures 
          gpu **
          elems_tile |-> Frac (1.0R /. p.blockWidth) v_elems **
          col_ind_tile |-> Frac (1.0R /. p.blockWidth) v_col_ind **
          gB |-> Frac (fB /. nthreads p) eB **
          live out **
          (exists* (k_v x_v : sz).
            k |-> k_v **
            x |-> x_v **
            pure (
              k_v < p.blockItemsK /\
              x_v < p.blockItemsX /^ p.blockWidth
            )
          )
      {
        let b = M.gpu_matrix_read gB c dense_off;
        open Pulse.Lib.Array;
        Pulse.Lib.Array.pts_to_len out;
        let c = out.(!x);
        out.(!x) <- (c `add` (a `mul` b));
        ();
      };

      x := !x +^ 1sz;
    };

    k := !k +^ 1sz;
  };
}

// TODO implementar optimización para unrollear loops
inline_for_extraction noextract
fn compute_residue
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
  (#v_elems : erased (seq et))
  (#v_col_ind : erased (seq sz))
  (#_ : squash(forall i. 0 <= v_col_ind @! i /\ v_col_ind @! i < p.shared))
  // matriz densa B
  (#eB : erased (ematrix et p.shared p.cols))
  // resultado parcial
  (#v_out : erased (seq et))
  (bid : szlt (nblocks p))
  (tid : szlt p.blockWidth)
  (n_idx : szlt p.cols {SZ.v n_idx == bcol p bid})
  (residue : szlt p.blockItemsK)
  norewrite
  preserves
    gpu **
    gpu_pts_to_slice elems_tile #(1.0R /. p.blockWidth)
      0 residue v_elems **
    gpu_pts_to_slice col_ind_tile #(1.0R /. p.blockWidth)
      0 residue v_col_ind **
    gB |-> Frac (fB /. nthreads p) eB
  requires
    pure (fits (p.cols + p.blockItemsX)) **
    pure (len v_elems == residue /\ len v_col_ind == residue) **
    out |-> v_out
  ensures
    //out |-> v_out + dprod v_elems v_col_ind eB (col := tid)
    live out
{
  Pulse.Lib.Array.pts_to_len out;

  let mut k : sz = 0sz;
  while (!k <^ residue)
    invariant
      // TODO decir algo sobre el producto
      live out ** live k **
      pure (
        !k <= residue
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
          !k < residue /\
          !x <= p.blockItemsX /^ p.blockWidth
        )
    {
      block_lemma_off p.blockItemsX p.blockWidth !x tid;

      let dense_off : sz = n_idx +^ !x *^ p.blockWidth +^ tid;

      if (dense_off <^ p.cols)
        ensures 
          gpu **
          gpu_pts_to_slice elems_tile #(1.0R /. p.blockWidth)
            0 residue v_elems **
          gpu_pts_to_slice col_ind_tile #(1.0R /. p.blockWidth)
            0 residue v_col_ind **
          gB |-> Frac (fB /. nthreads p) eB **
          live out **
          (exists* (k_v x_v : sz).
            k |-> k_v **
            x |-> x_v **
            pure (
              k_v < residue /\
              x_v < p.blockItemsX /^ p.blockWidth
            )
          )
      {
        let b = M.gpu_matrix_read gB c dense_off;
        open Pulse.Lib.Array;
        Pulse.Lib.Array.pts_to_len out;
        let c = out.(!x);
        out.(!x) <- (c `add` (a `mul` b));
        ();
      };

      x := !x +^ 1sz;
    };

    k := !k +^ 1sz;
  };
}

ghost
fn when__intro_true (b:bool{b == true}) (p : slprop)
  requires p
  ensures when__ b (fun _ -> p)
{
  rewrite p as when__ b (fun _ -> p);
}

ghost
fn when__intro_false (b : bool{b == false}) (p : slprop)
  ensures when__ b (fun _ -> p)
{
  rewrite emp as when__ b (fun _ -> p);
}

ghost
fn when__elim_true (b:bool{b == true}) (p : slprop)
  requires when__ b (fun _ -> p)
  ensures p
{
  rewrite when__ b (fun _ -> p) as p;
}

ghost
fn when__elim_false (b:bool{b == false}) (p : slprop)
  requires when__ b (fun _ -> p)
  ensures emp
{
  rewrite when__ b (fun _ -> p) as emp;
}

inline_for_extraction noextract
fn store_out
  (#et : Type0) {| scalar et |}
  (p : parameters)
  // TODO esto es raro acá
  (#_ : squash (fits (p.cols + p.blockItemsX)))
  (#lC : mlayout p.rows p.cols)
  {| clayout lC |}
  (gC : M.gpu_matrix et lC)
  (out : larray et (p.blockItemsX /^ p.blockWidth))
  (#v_out : erased (seq et){length out == len v_out})
  (bid : szlt (nblocks p))
  (tid : szlt p.blockWidth)
  (m_idx : szlt p.rows{SZ.v m_idx == brow p bid})
  (n_idx : szlt p.cols{SZ.v n_idx == bcol p bid})
  (x : szlt (p.blockItemsX /^ p.blockWidth))
  requires
    when__ (bcol p bid + x * p.blockWidth + tid < p.cols) (fun _ ->
      matrix_live_cell gC (brow p bid) (bcol p bid + x * p.blockWidth + tid)
    )
    ** out |-> v_out
  ensures
    when__ (bcol p bid + x * p.blockWidth + tid < p.cols) (fun _ ->
        M.gpu_matrix_pts_to_cell gC
          (brow p bid)
          (bcol p bid + x * p.blockWidth + tid)
          (v_out @! x)
    )
    ** out |-> v_out
{
  block_lemma_off p.blockItemsX p.blockWidth x tid;

  let out_off = n_idx +^ x *^ p.blockWidth +^ tid;
  assert rewrites_to out_off (n_idx +^ x *^ p.blockWidth +^ tid);

  if (out_off <^ p.cols) {
    when__elim_true _ _;
    unfold matrix_live_cell;

    open Pulse.Lib.Array;
    let c = out.(x);
    assert pure (n_idx +^ x *^ p.blockWidth +^ tid <^ p.cols);

    assert rewrites_to #sz m_idx (SZ.uint_to_t (brow p bid));
    assert rewrites_to #sz n_idx (SZ.uint_to_t (bcol p bid));

    M.gpu_matrix_write_cell gC m_idx (n_idx +^ x *^ p.blockWidth +^ tid) c;

    assert M.gpu_matrix_pts_to_cell gC (brow p bid) (bcol p bid + x * p.blockWidth + tid) (v_out @! x);
    when__intro_true (bcol p bid + x * p.blockWidth + tid < p.cols)
      (M.gpu_matrix_pts_to_cell gC (brow p bid) (bcol p bid + x * p.blockWidth + tid) (v_out @! x));
  }
  else {
    rewrite when__ (bcol p bid + x * p.blockWidth + tid < p.cols) (fun _ ->
      matrix_live_cell gC (brow p bid) (bcol p bid + x * p.blockWidth + tid)
    ) as when__ (bcol p bid + x * p.blockWidth + tid < p.cols) (fun _ ->
      M.gpu_matrix_pts_to_cell gC
        (brow p bid) (bcol p bid + x * p.blockWidth + tid)
        (v_out @! x)
    );

    // por que no anda?
    // when__elim_false _ _;
    //assert pure (bcol p bid + x * p.blockWidth + tid < p.cols == false);
    //when__intro_false (bcol p bid + x * p.blockWidth + tid < p.cols)
      //(m.gpu_matrix_pts_to_cell gc (brow p bid) (bcol p bid + x * p.blockWidth + tid) (v_out @! x));
  };

}


#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (p : parameters)
  (#lb : mlayout p.shared p.cols)
  (#lc : mlayout p.rows p.cols)
  {| clayout lb, clayout lc |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (gB : M.gpu_matrix et lb)
  (gC : M.gpu_matrix et lc)
  // matriz sparse ga
  (#elems : lseq et gA.nnz)
  (#col_ind : lseq sz gA.nnz)
  (#row_off : lseq sz (p.rows + 1))
  (#eA : ematrix et p.rows p.shared)
  // matriz densa gb
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
  let m_idx = brow_ p bid;
  let n_idx = bcol_ p bid;

  let (elems_tile, (col_ind_tile, _)) = sh;

  assert rewrites_to elems_tile (fst sh);
  assert rewrites_to col_ind_tile (fst (snd sh));

  gpu_pts_to_ref elems_tile;
  gpu_pts_to_ref col_ind_tile;

  let ri = gpu_array_read gA.row_off m_idx;
  let re = gpu_array_read gA.row_off (m_idx +^ 1sz);

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
      elems_tile col_ind_tile bid ri re !idx tid #();
 
    compute p elems_tile col_ind_tile gB out bid tid n_idx;

    idx := !idx +^ 1sz;
    nnz := !nnz -^ p.blockItemsK;
  };

  assert pure (ri + !idx * p.blockItemsK <= re);
  assert pure (re - (ri + !idx * p.blockItemsK) < p.blockItemsK);
  sparse_load_residue p gA #row_off #elems #col_ind #eA
    elems_tile col_ind_tile bid ri re !idx tid;
  compute_residue p elems_tile col_ind_tile gB out bid tid n_idx
    (re -^ (ri +^ !idx *^ p.blockItemsK));
  
  
  unfold slice_live elems_tile #(1.0R /. p.blockWidth)
    (re - (ri + !idx * p.blockItemsK)) p.blockItemsK;
  gpu_slice_concat elems_tile #(1.0R /. p.blockWidth)
    0 (re - (ri + !idx * p.blockItemsK)) p.blockItemsK;

  unfold slice_live col_ind_tile #(1.0R /. p.blockWidth)
    (re - (ri + !idx * p.blockItemsK)) p.blockItemsK;
  gpu_slice_concat col_ind_tile #(1.0R /. p.blockWidth)
    0 (re - (ri + !idx * p.blockItemsK)) p.blockItemsK;

  with v_out. assert out |-> v_out;
  Pulse.Lib.Array.pts_to_len out;


  foreach (p.blockItemsX /^ p.blockWidth)
    (fun x ->
      when__ (bcol p bid + x * p.blockWidth + tid < p.cols) 
        (fun _ -> matrix_live_cell gC (brow p bid) (bcol p bid + x * p.blockWidth + tid)))
    (fun x ->
      when__ (bcol p bid + x * p.blockWidth + tid < p.cols) 
        (fun _ ->
          M.gpu_matrix_pts_to_cell gC
            (brow p bid)
            (bcol p bid + x * p.blockWidth + tid)
            (v_out @! x)))
    (store_out p gC out bid tid m_idx n_idx);

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
#pop-options


inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (p : parameters{size_req p})
  (#lB : mlayout p.shared p.cols)
  (#lC : mlayout p.rows p.cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared){is_global_smatrix gA})
  (gB : M.gpu_matrix et lB{M.is_global_matrix gB})
  (gC : M.gpu_matrix et lC{M.is_global_matrix gC})
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (#eB : ematrix et p.shared p.cols)
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  : kernel_desc
    (
      smatrix_pts_to' gA #fA elems col_ind row_off eA **
      gB |-> Frac fB eB **
      live gC
    )
    (
      smatrix_pts_to' gA #fA elems col_ind row_off eA **
      gB |-> Frac fB eB **
      gC |-> MS.matmul eA eB
    )
= admit();{
  nblk = sz_nblocks p;
  nthr = sz_nthreads p;

  barrier_contract = (fun bid ptrs ->
    barrier_contract p elems col_ind row_off
      (fst ptrs) (fst (snd ptrs)) bid);
  barrier_ok = (fun bid ptrs -> magic());

  shmems_desc = shmems_desc et p;

  frame = emp;

  block_pre  = (fun bid -> forall+ (tid : natlt p.blockWidth).
    block_pre p gA gB gC elems col_ind row_off eA eB fA fB bid tid);
  block_post = (fun bid -> forall+ (tid : natlt p.blockWidth).
    block_post p gA gB gC elems col_ind row_off eA eB fA fB bid tid);
  setup    = setup    p gA gB gC elems col_ind row_off #_ #_ #fA;
  teardown = teardown p gA gB gC elems col_ind row_off #_ #_ #fA;

  block_frame    = (fun _ar _bid -> emp);
  block_setup    = block_setup    p gA gB gC elems col_ind row_off #eA #eB #fA #fB;
  block_teardown = block_teardown p gA gB gC elems col_ind row_off #eA #eB #fA #fB;

  kpre  = kpre  p gA gB gC elems col_ind row_off eA eB fA fB;
  kpost = kpost p gA gB gC elems col_ind row_off eA eB fA fB;

  f = kf p gA gB gC;

  block_pre_sendable=solve;
  block_post_sendable=solve;
  kpre_sendable=magic();
  kpost_sendable=magic();
}

inline_for_extraction noextract
fn spmm
  (#et : Type0) {| scalar et |}
  (rows shared cols : szp)
  (blockItemsK : szp)
  (blockItemsX : szp)
  (blockWidth : (k : szp {k /? blockItemsK /\ k /? blockItemsX}))
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| cB : clayout lB, cC : clayout lC |}
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
  //(#_ : size_req rows shared cols)
  norewrite
  preserves
    cpu **
    //on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (smatrix_pts_to' gA #fA elems col_ind row_off eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (blockItemsX /? cols) **
    on gpu_loc (live gC) **
    pure (rows * cols / blockItemsX <= max_blocks) **
    pure (blockWidth <= max_threads)
  ensures on gpu_loc (gC |-> MS.matmul eA eB)
{
  let params = { rows; shared; cols; blockItemsK; blockItemsX; blockWidth };
  assume pure (well_formed params #gA.nnz col_ind row_off);
  // que raro
  let pf_size_req : squash (size_req params) = ();
  launch_sync (
    kdesc #et #_ params #lB #lC #cB #cC
      gA gB gC elems col_ind row_off eA
      #eB #fA #fB
  );
}

let _spmm_u32 (rows shared cols : szp)
  (#_ : squash (SZ.fits (rows * cols) /\ SZ.fits (shared * cols)))
  =
  spmm #u32 #_ rows shared cols 128sz 128sz 32sz #(row_major _ _) #(row_major _ _)
    #_ #_
