module Kuiper.Sparse.SPMM

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


open Kuiper.Poly.GEMMGPU.Type

inline_for_extraction noextract
let size_req : size_req_t =
  fun mrows mshared mcols ->
    mrows <= max_blocks /\
    mcols <= max_threads

(* Description of shared memory used in this kernel. *)
inline_for_extraction noextract
let shmems_desc (et:Type0) {| sized et |}
  (blockItemsK : szp) : list shmem_desc = [
  SHArray et blockItemsK;
  // TODO podemos parametrizar este tipo?
  SHArray u32 blockItemsK;
]

unfold
let barrier_pre
  (#rows #shared #cols : nat)
  (#nnz : sz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (rows + 1))
  (blockItemsK : szp)
  (bid : natlt rows)
  : prop
  =
    (
      let ri = row_off @! bid in
      let re = row_off @! bid + 1 in
      // condicion de smatrix
      valid_smatrix rows shared (cast_pos col_ind) (cast_pos row_off) /\
     // TODO quitar estas condiciones
      // esta dice que no hay residuo
      blockItemsK /? (re - ri) /\
      // esta dice que cada thread no es responsable de mas de un valor
      SZ.v blockItemsK <= cols /\
      // esta dice que no hace falta enmascarar
      cols /? SZ.v blockItemsK
    )


let barrier_p
  (#et : Type0)
  (#rows #shared #cols : nat)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (rows + 1))
  (#blockItemsK : szp)
  (elems_tile : gpu_array et blockItemsK)
  (col_ind_tile : gpu_array u32 blockItemsK)
  (bid : natlt rows)
  : Pure (B.barrier_side cols)
    (requires barrier_pre #rows #shared #cols col_ind row_off blockItemsK bid)
    (ensures fun _ -> true)
  =
  fun it tid ->
    let ri = row_off @! bid in
    let re = row_off @! bid + 1 in
    // nos pasamos
    if (it / 2) * blockItemsK >= re - ri then emp else 
    let off = ri + (it / 2) * blockItemsK in
    let u32_col_ind = Seq.map_seq sizet_to_u32 col_ind in
    let u32_col_ind = Seq.map_seq sizet_to_u32 col_ind in
    // a veces no puede probar esto
    assume (it / 2) * blockItemsK + tid < re - ri; 
    if even it then
      (exists* (s : seq et). elems_tile |-> Frac (1.0R /. cols) s) **
      (exists* (s : seq u32). col_ind_tile |-> Frac (1.0R /. cols) s)
    else
      gpu_pts_to_cell elems_tile tid (elems @! off + tid) **
      gpu_pts_to_cell col_ind_tile tid (u32_col_ind @! off + tid)

let barrier_q
  (#et : Type0)
  (#rows #shared #cols : nat)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (rows + 1))
  (#blockItemsK : szp)
  (elems_tile : gpu_array et blockItemsK)
  (col_ind_tile : gpu_array u32 blockItemsK)
  (bid : natlt rows)
  : Pure (B.barrier_side cols)
    (requires barrier_pre #rows #shared #cols col_ind row_off blockItemsK bid)
    (ensures fun _ -> true)
  =
  fun it tid ->
    let ri = row_off @! bid in
    let re = row_off @! bid + 1 in
    // nos pasamos
    if (it / 2) * blockItemsK >= re - ri then emp else 
    let off = ri + (it / 2) * blockItemsK in
    let u32_col_ind = Seq.map_seq sizet_to_u32 col_ind in
    // a veces no puede probar esto
    assume (it / 2) * blockItemsK + tid < re - ri; 
    if even it then
      (exists* (x : et). gpu_pts_to_cell elems_tile tid x) **
      (exists* (x : u32). gpu_pts_to_cell col_ind_tile tid x)
    else
      elems_tile |-> Frac (1.0R /. cols) (Seq.slice elems off (off + blockItemsK)) **
      col_ind_tile |-> Frac (1.0R /. cols) (Seq.slice u32_col_ind off (off + blockItemsK)) 

unfold
let kpre1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : nat)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : smatrix et rows shared)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
  (eA : ematrix et rows shared)
  // matrices densas
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (fA fB : perm)
  (bid : natlt rows)
  (tid : natlt cols)
  : slprop
  =
  smatrix_pts_to' gA #(fA /. (rows * cols))
    elems col_ind row_off eA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  M.gpu_matrix_pts_to_cell gC bid tid
    (macc eC bid tid)

unfold
let kpost1
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : nat)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : smatrix et rows shared)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
  (eA : ematrix et rows shared)
  // matrices densas
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (fA fB : perm)
  (bid : natlt rows)
  (tid : natlt cols)
  : slprop
  =
  smatrix_pts_to' gA #(fA /. (rows * cols))
    elems col_ind row_off eA **
  gB |-> Frac (fB /. (rows * cols)) eB **
  M.gpu_matrix_pts_to_cell gC bid tid
    (MS.matmul_single  eA eB bid tid)


let barrier_tok
  (#et : Type0)
  (#rows #shared #cols : nat)
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (rows + 1))
  (#blockItemsK : szp)
  (elems_tile : gpu_array et blockItemsK)
  (col_ind_tile : gpu_array u32 blockItemsK)
  (bid : natlt rows)
  (it : nat)
  : Ghost slprop
    (requires barrier_pre #rows #shared #cols col_ind row_off blockItemsK bid)
    (ensures fun _ -> true)
  =
  B.barrier_tok
    (barrier_p  #_ #rows #shared #cols elems col_ind row_off elems_tile col_ind_tile bid)
    (barrier_q #_ #rows #shared #cols elems col_ind row_off elems_tile col_ind_tile bid)

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : nat)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : smatrix et rows shared)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
  (eA : ematrix et rows shared)
  // matrices densas
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (fA fB : perm)
  (#blockItemsK : szp)
  (sh : c_shmems (shmems_desc et blockItemsK))
  (bid : natlt rows)
  (tid : natlt cols)
  : Ghost (slprop)
    (requires barrier_pre #rows #shared #cols col_ind row_off blockItemsK bid)
    (ensures fun _ -> true)
  =
  let (elems_tile, (col_ind_tile, _)) = sh in
  kpre1 comb gA gB gC elems col_ind row_off eA eB eC fA fB bid tid **
  (exists* (s : seq et). elems_tile |-> Frac (1.0R /. cols) s) **
  (exists* (s : seq u32). col_ind_tile |-> Frac (1.0R /. cols) s) **
  barrier_tok #_ #rows #shared #cols
    elems col_ind row_off elems_tile col_ind_tile bid tid **
    B.barrier_state 0

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : nat)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  (gA : smatrix et rows shared)
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
  (eA : ematrix et rows shared)
  // matrices densas
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (fA fB : perm)
  (#blockItemsK : szp)
  (sh : c_shmems (shmems_desc et blockItemsK))
  (bid : natlt rows)
  (tid : natlt cols)
  : Ghost (slprop)
    (requires barrier_pre #rows #shared #cols col_ind row_off blockItemsK bid)
    (ensures fun _ -> true)
  =
  let (elems_tile, (col_ind_tile, _)) = sh in
  let ri = row_off @! bid in
  let re = row_off @! bid + 1 in
  kpost1 comb gA gB gC elems col_ind row_off eA eB eC fA fB bid tid **
  live elems_tile #(1.0R /. cols) **
  live col_ind_tile #(1.0R /. cols) **
  barrier_tok #_ #rows #shared #cols
    elems col_ind row_off (elems_tile) (col_ind_tile)
    bid  tid **
    B.barrier_state (2 * (re - ri) / blockItemsK)

#set-options "--debug SMTFail --split_queries always"
//#set-options "--print_implicits"
inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (comb : binop et)
  (#rows #shared #cols : SZ.t)
  (#lB : mlayout shared cols)
  (#lC : mlayout rows cols)
  {| clayout lB, clayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared))
  (gB : M.gpu_matrix et lB)
  (gC : M.gpu_matrix et lC)
  // matriz sparse gA
  (#elems : lseq et gA.nnz)
  (#col_ind : lseq sz gA.nnz)
  (#row_off : lseq sz (rows + 1))
  (#eA : ematrix et rows shared)
  // matriz densa gB
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
  (#fA #fB : perm)
  (#blockItemsK : szp)
  (sh : c_shmems (shmems_desc et blockItemsK))
  (bid : szlt rows)
  (tid : szlt cols)
  // TODO esto está bien?
  (_ : squash (barrier_pre #rows #shared #cols col_ind row_off blockItemsK bid))
  norewrite
  requires
    gpu **
    kpre comb gA gB gC elems col_ind row_off eA eB eC fA fB sh bid tid **
    thread_id cols tid **
    block_id rows bid
  ensures
    gpu **
    kpost comb gA gB gC elems col_ind row_off eA eB eC fA fB sh bid tid **
    thread_id cols tid **
    block_id rows bid
{
  let trow = bid; assert (rewrites_to trow bid);
  let tcol = tid; assert (rewrites_to tcol tid);

  // let m_idx = trow;
  // let n_idx = 0;
  // let blockItemsX = cols;

  let (elems_tile, (col_ind_tile, _)) = sh;

  gpu_pts_to_ref elems_tile;
  gpu_pts_to_ref col_ind_tile;

  let ri = gpu_array_read gA.row_off trow;
  let re = gpu_array_read gA.row_off (trow +^ 1sz);

  let mut dp : et = zero;
  let mut nnz = re -^ ri;
  let mut idx = 0sz;

  unfold barrier_tok #_ #rows #shared #cols
    elems col_ind row_off elems_tile col_ind_tile bid tid;

  while ((!nnz >=^ blockItemsK))
    invariant
      live dp ** // TODO decir algo sobre el producto
      live nnz **
      live idx **
      B.barrier_state (!idx * 2) **
      (exists* (s : seq et). elems_tile |-> Frac (1.0R /. cols) s) **
      (exists* (s : seq u32). col_ind_tile |-> Frac (1.0R /. cols) s) **
      pure (
        !idx <= (re - ri) / blockItemsK /\
        !nnz == re -^ ri -^ !idx *^ blockItemsK
      ) 
  {
    rewrite 
      (exists* (s : seq et). elems_tile |-> Frac (1.0R /. cols) s) **
      (exists* (s : seq u32). col_ind_tile |-> Frac (1.0R /. cols) s)
      as barrier_p 
        #et #rows #shared #cols #gA.nnz
        elems col_ind row_off elems_tile col_ind_tile bid (!idx * 2) tid;

    B.barrier_wait ();
    admit();

    rewrite
      (barrier_q #et #rows #shared #cols #gA.nnz
        elems col_ind row_off
        #blockItemsK elems_tile col_ind_tile
        bid (!idx * 2) tid)
      as (exists* (x : et). gpu_pts_to_cell elems_tile tid x) **
        (exists* (c : u32). gpu_pts_to_cell col_ind_tile tid c);

    let x = gpu_array_read elems (!idx * blockItemsK + tid);
    gpu_array_write elems_tile tid x;

    let c = sizet_to_u32 (gpu_array_read col_ind (!idx * blockItemsK + tid));
    gpu_array_write col_ind_tile tid c;

    let u32_col_ind = Seq.map_seq sizet_to_u32 col_ind;

    rewrite 
      gpu_pts_to_cell elems_tile tid (elems @! ri + !idx * blockItemsK + tid) **
      gpu_pts_to_cell col_ind_tile tid
        (u32_col_ind @! ri + !idx * blockItemsK + tid)
      as barrier_p 
        #et #rows #shared #cols #gA.nnz
        elems col_ind row_off elems_tile col_ind_tile bid (!idx * 2 + 1) tid;

    B.barrier_wait ();

    rewrite
      (barrier_q #et #rows #shared #cols #gA.nnz
        elems col_ind row_off
        #blockItemsK elems_tile col_ind_tile
        bid (!idx * 2 + 1) tid)
      as
        elems_tile |-> Frac (1.0R /. cols)
          (Seq.slice elems (ri + !idx * blockItemsK) (ri + !idx * blockItemsK + blockItemsK)) **
        col_ind_tile |-> Frac (1.0R /. cols)
          (Seq.slice u32_col_ind (ri + !idx * blockItemsK) (ri + !idx * blockItemsK + blockItemsK));


    // cargamos la matriz densa y hacemos el producto en una pasada
    let mut k : sz = 0sz;
    while ((!k < blockItemsK))
      invariant
        live dp ** // TODO decir algo sobre el producto
        live idx **
        live k **
        pure (
          !k <= blockItemsK /\
          (!k < blockItemsK ==> !idx * !k < re - ri) // hace falta?
        )
    {
      let x = gpu_array_read elems_tile !k;
      let c = gpu_array_read col_ind_tile !k;
      //let y = M.gpu_matrix_read gB c (n_idx + tid);
      let y = M.gpu_matrix_read gB c tid;
      dp := !dp `add` (x `mul` y);

      k := !k +^ 1sz;
    };

    idx := !idx +^ 1sz;
    nnz := !nnz -^ blockItemsK;
  };

  M.gpu_matrix_write_cell gC trow tcol !dp;

  admit();
        
}
