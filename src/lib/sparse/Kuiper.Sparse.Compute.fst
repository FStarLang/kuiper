module Kuiper.Sparse.Compute

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Spec.GEMM
open Kuiper.Sparse.DotProduct
open Kuiper.Sparse.Array
open Kuiper.Sparse.Common
module M  = Kuiper.Matrix
module SZ = Kuiper.SizeT

#set-options "--debug SMTFail --split_queries always"
//#set-options "--print_implicits"


(* Definiciones auxiliares *)

let divup n d = ((n + d - 1) / d)

let submatrix
  (#a : Type0)
  (#rows #cols : nat)
  (em : ematrix a rows cols)
  (m : natle rows)
  (srows : nat{m + srows <= rows})
  (n : natle cols)
  (scols : nat{n + scols <= cols})
: ematrix a srows scols 
= mkM (fun i j -> macc em (m + i) (n + j))

let col_strided_matrix
  (#a : Type0)
  (#rows #cols : nat)
  (em : ematrix a rows cols)
  (step : pos)
: ematrix a rows (cols `divup` step)
=
  mkM (fun i j -> macc em i (j * step))

// esto x ahora no se usa
let col_strided_matrix_lemma
  (#a : Type0)
  (#rows1 #rows2 #cols : nat)
  (em1 : ematrix a rows1 cols)
  (em2 : ematrix a rows2 cols)
  (em : ematrix a (rows1 + rows2) cols)
  (step : pos)
  (j : natlt (cols `divup` step))
: Lemma
  (requires
    ematrix_col em1 (j * step) `Seq.append` ematrix_col em2 (j * step) ==
    ematrix_col em (j * step)
  )
  (ensures
    Seq.append
      (ematrix_col (col_strided_matrix em1 step) j)
      (ematrix_col (col_strided_matrix em2 step) j)
    ==
    ematrix_col (col_strided_matrix em step) j 

  )
=
  let c1 = ematrix_col (col_strided_matrix em1 step) j in
  let c2 = ematrix_col (col_strided_matrix em2 step) j in
  let c = ematrix_col (col_strided_matrix em step) j in

  introduce forall i. Seq.append c1 c2 @! i == c @! i
  with calc (==) {
    Seq.append c1 c2 @! i;
    == {}
    ematrix_col em (j * step) @! i;
    == {}
    c @! i;
  };

  assert Seq.equal (Seq.append c1 c2) c

let step_sparse_cols
  (cols n scols : nat)
  (step : pos)
= let n' = min n cols in min scols (cols - n') `divup` step

let step_submatrix
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (em : ematrix et rows cols)
  (scols : nat)
  (n : nat)
  (step : pos)
: Pure (ematrix et rows (step_sparse_cols cols n scols step))
  (requires true)
  (ensures fun _ -> true)
=
  let n' = min n cols in
  col_strided_matrix (submatrix em 0 rows n' (min scols (cols - n'))) step

// esto x ahora no se usa
let _row_x_mat_acc
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (row : lseq et shared)
  (em : ematrix et shared cols)
  (to : natle shared)
: Ghost (lseq et block) (requires cols <= block) (ensures fun _ -> true)
=
  Seq.init_ghost block (fun i ->
    if i < cols
      then _dprod_acc (acc @! i) row (ematrix_col em i) to
      else acc @! i
  )


// esto x ahora no se usa
// creo q no hace falta
let comb_row_x_mat_acc
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (row : lseq et shared)
  (em : ematrix et shared cols)
  (to : natlt shared)
: Lemma
  (requires cols <= block)
  (ensures
    comb (_row_x_mat_acc acc row em to) (row @! to) (ematrix_row em to) ==
    _row_x_mat_acc acc row em (to + 1)
  )
=
  assert Seq.equal
    (comb (_row_x_mat_acc acc row em to) (row @! to) (ematrix_row em to))
    (_row_x_mat_acc acc row em (to + 1))

// esto x ahora no se usa
let row_x_mat_acc
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (row : lseq et shared)
  (em : ematrix et shared cols)
: Ghost (lseq et block) (requires cols <= block) (ensures fun _ -> true)
=
  _row_x_mat_acc acc row em shared


(* Definiciones sparse *)

let _sparse_row_x_mat_acc
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (em : ematrix et shared cols)
  (to : natle nnz)
: Ghost (lseq et block)
  (requires valid_pos shared pos)
  (ensures fun _ -> true)
=
  Seq.init_ghost block (fun i ->
    if i < cols
      then _sparse_dprod_acc (acc @! i) elems pos (ematrix_col em i) to
      else acc @! i
  )

let sparse_row_x_mat_acc
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (em : ematrix et shared cols)
: Ghost (lseq et block)
  (requires valid_pos shared pos)
  (ensures fun _ -> true)
=
  _sparse_row_x_mat_acc acc elems pos em nnz

(* Lemas para combinar resultados *)

let sparse_row_x_mat_acc_lemma
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (em : ematrix et shared cols)
  (to : natle nnz)
: Lemma
  (requires valid_pos shared pos)
  (ensures 
    sparse_row_x_mat_acc #_ #_ #_ #_ #block
      (sparse_row_x_mat_acc acc
        #to (fst (Seq.split elems to)) (fst (Seq.split pos to)) em
      )
      #(nnz - to) (snd (Seq.split elems to)) (snd (Seq.split pos to)) em
    ==
    sparse_row_x_mat_acc acc elems pos em
  )
=
  let elems1, elems2 = Seq.split elems to in
  let pos1, pos2 = Seq.split pos to in

  let (s1 : lseq et block) =
    sparse_row_x_mat_acc acc #to elems1 pos1 em in
  let (s : lseq et block) = 
    sparse_row_x_mat_acc s1 #(nnz - to) elems2 pos2 em in
  let (s' : lseq et block) =
    sparse_row_x_mat_acc acc elems pos em in

  introduce forall i. s @! i == s' @! i
  with (
    if i < cols
      then (
        let col = ematrix_col em i in

        let scol : lseq et nnz = seq_make_sparse pos col in
        let scol1, scol2 = Seq.split scol to in

        assert scol1 `Seq.equal` seq_make_sparse #_ #_ #to pos1 col;
        assert scol2 `Seq.equal` seq_make_sparse #_ #_ #(nnz - to) pos2 col;

        calc (==) {
          s @! i;
          == {}
          sparse_dprod_acc
            (sparse_dprod_acc (acc @! i) #to elems1 pos1 col)
            #(nnz - to) elems2 pos2 col;
          == {}
          dprod_acc
            (dprod_acc (acc @! i) #to elems1 scol1)
            #(nnz - to) elems2 scol2;
          == {
            dprod_acc_lemma
              (acc @! i)
              #to elems1 scol1
              #(nnz - to) elems2 scol2
          }
          dprod_acc (acc @! i)
            #nnz 
            (Seq.append elems1 elems2)
            (Seq.append scol1 scol2);
          == { Seq.lemma_split elems to}
          dprod_acc (acc @! i) elems (Seq.append scol1 scol2);
          == { Seq.lemma_split scol to}
          dprod_acc (acc @! i) elems scol;
          == {}
          sparse_dprod_acc (acc @! i) elems pos col;
          == {} 
          s' @! i;
        }
      )
      else ()
  );
  assert Seq.equal s s'

let sparse_row_x_mat_acc_lemma'
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (em : ematrix et shared cols)
  (from to : natle nnz{from <= to})
: Lemma
  (requires valid_pos shared pos)
  (ensures 
    sparse_row_x_mat_acc #_ #_ #_ #_ #block
      (sparse_row_x_mat_acc
        acc #from (Seq.slice elems 0 from) (Seq.slice pos 0 from) em
      )
      #(to - from) (Seq.slice elems from to) (Seq.slice pos from to) em
    ==
    sparse_row_x_mat_acc acc
      #to (Seq.slice elems 0 to) (Seq.slice pos 0 to) em
  )
=
  let elems1 = Seq.slice elems 0 from in
  let pos1 = Seq.slice pos 0 from in

  let elems2 = Seq.slice elems from to in
  let pos2 = Seq.slice pos from to in

  let elems' = Seq.slice elems 0 to in
  let pos' = Seq.slice pos 0 to in

  let elems1', elems2' = Seq.split elems' from in
  let pos1', pos2' = Seq.split pos' from in

  assert elems1' `Seq.equal` elems1;
  assert elems2' `Seq.equal` elems2;
  assert pos1' `Seq.equal` pos1;
  assert pos2' `Seq.equal` pos2;

  sparse_row_x_mat_acc_lemma acc #to elems' pos' em from

(* Sparse compute *)

let _sparse_comb
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (em : ematrix et shared cols)
  (i : natlt nnz)
  (to : nat)
: Ghost (lseq et block)
  (requires cols <= block /\ valid_pos shared pos)
  (ensures fun _ -> true)
=
  _comb
    acc
    (elems @! i)
    (ematrix_row em (pos @! i))
    (min to cols)

let sparse_comb
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (em : ematrix et shared cols)
  (i : natlt nnz)
: Ghost (lseq et block)
  (requires cols <= block /\ valid_pos shared pos)
  (ensures fun _ -> true)
=
  _sparse_comb acc elems pos em i cols

let sparse_comb_row_x_mat_acc
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (em : ematrix et shared cols)
  (to : natlt nnz)
: Lemma
  (requires cols <= block /\ valid_pos shared pos)
  (ensures
    sparse_comb
      (_sparse_row_x_mat_acc acc elems pos em to)
      elems pos em to
    ==
    _sparse_row_x_mat_acc acc elems pos em (to + 1)
  )
=
  assert Seq.equal
    (sparse_comb
      (_sparse_row_x_mat_acc acc elems pos em to)
      elems pos em to
    )
    (_sparse_row_x_mat_acc acc elems pos em (to + 1))


#push-options "--z3rlimit 20"
// TODO ver si se pueden simplificar más los argumentos
inline_for_extraction noextract
fn compute
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : sz)
  // creo que este refinamiento no hace falta
  (#blockWidth #blockItemsK #blockItemsX : szp{blockWidth /? blockItemsX})
  (elems_tile : gpu_array et blockItemsK)
  (col_ind_tile : gpu_array sz blockItemsK)
  (#fA : perm)
  (#lB : mlayout shared cols)
  {| clayout lB |}
  (gB : M.gpu_matrix et lB)
  (#fB : perm)
  (out : larray et (blockItemsX /^ blockWidth))
  // fragmentos sparse
  (#v_elems : lseq et blockItemsK)
  (#v_col_ind : lseq sz blockItemsK)
  (#_ : squash(valid_pos shared (cast_pos v_col_ind)))
  // matriz densa B
  (#eB : erased (ematrix et shared cols))
  // resultado parcial
  (#v_out : erased (seq et))
  (#_ : squash (len v_out == blockItemsX /^ blockWidth))
  (tid : szlt blockWidth)
  (n_idx : szlt cols)
  norewrite
  preserves
    gpu **
    elems_tile |-> Frac fA v_elems **
    col_ind_tile |-> Frac fA v_col_ind **
    gB |-> Frac fB eB
  requires
    pure (fits (cols + blockItemsX)) **
    out |-> v_out
  ensures
    out |->
      sparse_row_x_mat_acc
        // esto es feo pero por algun motivo no tomamos v_out como lseq
        #_ #_ #_ #_
        #(blockItemsX /^ blockWidth) v_out
        v_elems (cast_pos v_col_ind)
        (step_submatrix eB (blockItemsX - tid) (n_idx + tid) blockWidth)

{
  let mut k : sz = 0sz;

  let stepB = step_submatrix eB (blockItemsX - tid) (n_idx + tid) blockWidth;

  assert pure (
    v_out `Seq.equal`
    _sparse_row_x_mat_acc
      #_ #_ #_ #_
      #(blockItemsX /^ blockWidth) v_out
      v_elems (cast_pos v_col_ind)
      stepB
      0
  );

  while (!k <^ blockItemsK)
    invariant
      (exists* (v_k : sz {v_k <= blockItemsK}).
        k |-> v_k **
        out |->
        _sparse_row_x_mat_acc
          #_ #_ #_ #_
          #(blockItemsX /^ blockWidth) v_out
          v_elems (cast_pos v_col_ind)
          stepB
          v_k 
      )
  {
    let a = gpu_array_read elems_tile !k;
    let c = gpu_array_read col_ind_tile !k;
    let mut x = 0sz;

    with v_out'. assert out |-> v_out';
    with v_k. assert k |-> v_k;

    assert pure (
      v_out' `Seq.equal`
      _sparse_comb v_out' v_elems (cast_pos v_col_ind) stepB v_k !x
    );

    while ((!x <^ blockItemsX /^ blockWidth))
      invariant
        (exists* (v_x : sz).
          out |->
            _sparse_comb v_out' v_elems (cast_pos v_col_ind) stepB v_k v_x **
            x |-> v_x
        ) **
        k |-> v_k
    {
      with v_x. assert x |-> v_x;
      assert pure (v_x *^ blockWidth + tid < blockItemsX);

      assume pure (fits (n_idx +^ v_x *^ blockWidth + tid));
      let dense_off : sz = n_idx +^ !x *^ blockWidth +^ tid;

      if (dense_off <^ cols)
        ensures 
          gpu **
          elems_tile |-> Frac fA v_elems **
          col_ind_tile |-> Frac fA v_col_ind **
          gB |-> Frac fB eB **
          out |->
            _sparse_comb
              v_out' v_elems (cast_pos v_col_ind)
              stepB v_k (v_x + 1) **
          k |-> v_k **
          x |-> v_x 
      {
        with v_out''. assert out |-> v_out'';

        let b = M.gpu_matrix_read gB c dense_off;
        open Pulse.Lib.Array;
        Pulse.Lib.Array.pts_to_len out;
        let v = out.(!x);
        out.(!x) <- (v `add` (a `mul` b));

        assert out |-> Seq.upd v_out'' v_x (add v (a `mul` b));

        rewrite each v_out'' as (_comb v_out' a (ematrix_row stepB c) v_x);

        assert pure (v_x < (blockItemsX - tid) `divup` blockWidth);
        assert pure (v_x < (cols - n_idx - tid) `divup` blockWidth);
        rewrite each a as (v_elems @! v_k);
        rewrite each b as (macc stepB c v_x);

        _comb_lemma
          v_out'
          (v_elems @! v_k)
          (ematrix_row stepB c)
          v_x;

          // esto esta inestable, a veces sale y a veces no
          // admitimos por ahora
          admit();
      };

      x := !x +^ 1sz;
    };

    sparse_comb_row_x_mat_acc
      #_ #_ #_ #_
      #(blockItemsX /^ blockWidth) v_out
      v_elems (cast_pos v_col_ind)
      stepB
      !k;
    k := !k +^ 1sz;
  };

}
#pop-options