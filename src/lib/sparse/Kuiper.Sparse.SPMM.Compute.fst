module Kuiper.Sparse.SPMM.Compute

#lang-pulse

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Spec.GEMM
open Kuiper.Sparse.DotProduct
open Kuiper.Sparse.Array
open Kuiper.Sparse.Common
module M  = Kuiper.Matrix

(* Definiciones auxiliares *)

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

let step_submatrix_congr
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (em : ematrix et rows cols)
  (scols : nat)
  (n : nat)
  (step : pos)
  (x : natlt (step_sparse_cols cols n scols step))
: Lemma
  (requires true)
  (ensures
    ematrix_col em (min n cols + x * step) ==
    ematrix_col (step_submatrix em scols n step) x
  )
=
  assert Seq.equal
    (ematrix_col em (min n cols + x * step))
    (ematrix_col (step_submatrix em scols n step) x)


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

open Kuiper.Spec.GEMM

let sparse_row_x_mat_acc_congr
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (#block : nat)
  (acc : lseq et block)
  (#nnz : nat)
  (elems : lseq et nnz)
  (pos : lseq nat nnz)
  (em : ematrix et shared cols)
: Lemma
  (requires valid_pos shared pos /\ cols <= block)
  (ensures
    sparse_row_x_mat_acc acc elems pos em ==
    row_x_mat_acc acc (unsparse _ _ elems pos) em
  )
=
  let s = sparse_row_x_mat_acc acc elems pos em in
  let t = row_x_mat_acc acc (unsparse _ _ elems pos) em in

  introduce forall i. s @! i == t @! i
  with (
    if i < cols
      then sparse_dprod_acc_lemma (acc @! i) elems pos (ematrix_col em i)
      else ()
  );

  assert s `Seq.equal` t

let compute_result
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (bw bx : pos{bw /? bx})
  (#nnz : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz)
  (eB : ematrix et shared cols)
  (out : lseq et (bx / bw))
  (off : natlt bw)
  (n : natlt cols)
: Ghost (lseq et (bx / bw))
  (requires valid_pos shared col_ind)
  (ensures fun _ -> true)
=
  sparse_row_x_mat_acc
    out elems col_ind
    (step_submatrix eB (bx - off) (n + off) bw)

let compute_step
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (bw bx : pos{bw /? bx})
  (#nnz : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz)
  (eB : ematrix et shared cols)
  (out : lseq et (bx / bw))
  (off : natlt bw)
  (n : natlt cols)
  (from to : natle nnz{from <= to})
: Lemma
  (requires valid_pos shared col_ind)
  (ensures
    compute_result
      bw bx #(to - from)
      (Seq.slice elems from to) (Seq.slice col_ind from to) eB
      (compute_result
        bw bx #from
        (Seq.slice elems 0 from) (Seq.slice col_ind 0 from)
        eB out off n
      )
      off n ==
    compute_result
      bw bx #to
      (Seq.slice elems 0 to) (Seq.slice col_ind 0 to)
      eB out off n
  )
=
  sparse_row_x_mat_acc_lemma'
    out elems col_ind
    (step_submatrix eB (bx - off) (n + off) bw)
    from to

let compute_lemma
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : nat)
  (bw bx : pos{bw /? bx})
  (#nnz : nat)
  (elems : lseq et nnz)
  (col_ind : lseq nat nnz)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (out : lseq et (bx / bw))
  (off : natlt bw)
  (n : natlt cols)
  (i : natlt rows)
  (x : natlt (bx / bw))
: Lemma
  (requires
    valid_pos shared col_ind /\
    unsparse _ _ elems col_ind == ematrix_row eA i /\
    n + off + x * bw < cols /\
    forall i. out @! i == zero
  )
  (ensures
    compute_result bw bx elems col_ind eB out off n @! x ==
    matmul_single eA eB i (n + off + x * bw)
  )
=
  let sm = step_submatrix eB (bx - off) (n + off) bw in
  let j = n + off + x * bw in

  calc (==) {
    compute_result bw bx elems col_ind eB out off n @! x;
    == {}
    sparse_row_x_mat_acc out elems col_ind sm @! x;
    == { sparse_row_x_mat_acc_congr out elems col_ind sm }
    row_x_mat_acc out (unsparse _ _ elems col_ind) sm @! x;
    == {}
    row_x_mat_acc out (ematrix_row eA i) sm @! x;
    == {}
    dprod (ematrix_row eA i) (ematrix_col sm x);
    // == { assert ematrix_col sm x `Seq.equal` ematrix_col eB j}
    == { step_submatrix_congr eB (bx - off) (n + off) bw x }
    dprod (ematrix_row eA i) (ematrix_col eB j);
    == { dprod_is_matmul_single eA eB i j }
    matmul_single eA eB i j;
  }


#push-options "--z3rlimit 20"
// TODO ver si se pueden simplificar más los argumentos
inline_for_extraction noextract
fn compute
  (#et : Type0) {| scalar et |}
  (#shared #cols : sz)
  // creo que este refinamiento no hace falta
  (blockWidth blockItemsK blockItemsX : szp{blockWidth /? blockItemsX})
  // fragmentos sparse
  (elems_tile : gpu_array et blockItemsK)
  (col_ind_tile : gpu_array sz blockItemsK)
  (#fA : perm)
  (nnz : sz)
  (#v_elems : erased (lseq et nnz))
  (#v_col_ind : erased (lseq sz nnz))
  (#_ : squash(valid_pos shared (cast_pos v_col_ind)))
  // matriz densa B
  (#lB : mlayout shared cols)
  {| clayout lB |}
  (gB : M.gpu_matrix et lB)
  (#fB : perm)
  (#eB : erased (ematrix et shared cols))
  // resultado parcial
  (out : larray et (blockItemsX /^ blockWidth))
  (#v_out : erased (seq et))
  (#_ : squash (len v_out == blockItemsX /^ blockWidth))
  (tid : szlt blockWidth)
  (n_idx : szlt cols)
  norewrite
  preserves
    gpu **
    gpu_pts_to_slice elems_tile #fA 0 nnz v_elems **
    gpu_pts_to_slice col_ind_tile #fA 0 nnz v_col_ind **
    gB |-> Frac fB eB
  requires
    pure (fits (cols + blockItemsX)) **
    out |-> v_out
  ensures
    out |->
      compute_result
        blockWidth blockItemsX
        v_elems (cast_pos v_col_ind) eB v_out
        tid n_idx

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

  while (!k <^ nnz)
    invariant
      (exists* (v_k : sz {v_k <= nnz}).
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
          x |-> v_x **
          out |->
            _sparse_comb v_out' v_elems (cast_pos v_col_ind) stepB v_k v_x
        ) **
        k |-> v_k
    {
      with v_x. assert x |-> v_x;
      assert pure (v_x *^ blockWidth + tid < blockItemsX);

      // TODO arreglar
      assume pure (fits (n_idx +^ v_x *^ blockWidth + tid));
      let dense_off : sz = n_idx +^ !x *^ blockWidth +^ tid;

      if (dense_off <^ cols)
      {
        with v_out''. assert out |-> v_out'';

        let b = M.gpu_matrix_read gB c dense_off;
        open Pulse.Lib.Array;
        Pulse.Lib.Array.pts_to_len out;
        let v = out.(!x);
        out.(!x) <- (v `add` (a `mul` b));

        _comb_lemma
          v_out'
          (v_elems @! v_k)
          (ematrix_row stepB c)
          v_x;

        assert rewrites_to b (ematrix_row stepB c @! v_x);

        x := !x +^ 1sz;
      }
      else
      {
        x := !x +^ 1sz;
      };

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