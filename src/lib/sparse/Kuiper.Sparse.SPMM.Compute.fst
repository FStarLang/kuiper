module Kuiper.Sparse.SPMM.Compute

#lang-pulse

open Kuiper
#set-options "--z3rlimit 20"
module MU = Kuiper.Kernel.GEMM.Util
open Kuiper.Sparse.SPMM.Maps
open Kuiper.EMatrix
open Kuiper.Seq.Common { (@+), seq_replace }
open Kuiper.Spec.GEMM
open Kuiper.Array.Vectorized
open Kuiper.Tensor.Layout.Alg { l2_row_major, c_l2_row_major }
open Kuiper.Tensor.Layout.Slice
open Kuiper.Tensor
open Kuiper.Seq.Common { op_At_Bang }

open Kuiper.Sparse.Tensor
open Kuiper.Sparse.DotProduct
open Kuiper.Sparse.Array
open Kuiper.Sparse.Common
open Kuiper.Sparse.Approx
module Spec = Kuiper.Sparse.SPMM.Spec
open Kuiper.Sparse.FMA

module Math = Kuiper.Sparse.Math

(* Auxiliar lemmas for chests and sequences *)

// [up] on a 1-D concrete index is the corresponding abstract index.
let up_cidx1_eq (#d0:nat) (i:szlt d0)
  : Lemma (up (cidx1 i) == idx1 (v i))
          [SMTPat (up (cidx1 i))]
  = ()

// [seq_to_chest1] and [chest1_to_seq] are mutually inverse.
let chest1_to_seq_to_chest1 (#et : Type) (#n : nat) (s : lseq et n)
  : Lemma (chest1_to_seq (seq_to_chest1 s) == s)
          [SMTPat (chest1_to_seq (seq_to_chest1 s))]
  = Seq.lemma_eq_elim (chest1_to_seq (seq_to_chest1 s)) s

let seq_to_chest1_to_seq (#et : Type) (#n : nat) (c : chest1 et n)
  : Lemma (seq_to_chest1 (chest1_to_seq c) == c)
          [SMTPat (seq_to_chest1 (chest1_to_seq c))]
  = assert (equal (seq_to_chest1 (chest1_to_seq c)) c)

// The [lseq] view of a chest2 row is exactly [ematrix_row].
let chest2_row_to_seq (#et : Type0) (#rows #cols : nat)
  (em : chest2 et rows cols) (i : natlt rows)
  : Lemma (chest1_to_seq (chest2_row em i) == ematrix_row em i)
          [SMTPat (chest1_to_seq (chest2_row em i))]
  = Seq.lemma_eq_elim (chest1_to_seq (chest2_row em i)) (ematrix_row em i)

// The [lseq] view of a chest2 column is exactly [ematrix_col].
let ematrix_col_is_chest (#et : Type0) (#rows #cols : nat)
  (em : chest2 et rows cols) (j : natlt cols)
  : Lemma (ematrix_col em j == chest1_to_seq (chest2_col em j))
  = Seq.lemma_eq_elim (ematrix_col em j) (chest1_to_seq (chest2_col em j))

let chunk_end_bound (n k : nat) (ch : pos)
  : Lemma (requires ch /? n /\ ch /? k /\ k < n)
          (ensures k + ch <= n)
= lemma_divides_exact ch k;
  FStar.Math.Lemmas.swap_mul ch (k / ch);
  Math.block_lemma n ch (k / ch)

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn load_vmprod_chunk
  (#ta #tb #et : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb, sized tb, has_vec_cpy tb,
     scalar et, real_like et |}
  (mapA : ta -> et)
  (mapB : tb -> et)
  (rmapA : real -> real { MU.approx1 mapA rmapA })
  (rmapB : real -> real { MU.approx1 mapB rmapB })
  (ch : szp)
  (#n1 : sz { ch /? n1 })
  (y : larray et n1)
  (#vy : erased (lseq et n1))
  (x : ta)
  (#rx : erased real)
  (#ry : erased (lseq real n1))
  (#n2 : sz { ch /? n2 })
  (#lrow : layout1 n2) {| ctlayout lrow, clrow : cont_layout lrow |}
  (row : array1 tb lrow)
  (#frow : perm)
  (#vrow : chest1 tb n2)
  (#rrow : chest1 real n2)
  (k1 : sz { k1 + ch <= n1 })
  (k2 : sz { ch /? k2 })
  (bounds_checked : bool)
  preserves gpu
  preserves row |-> Frac frow vrow
  requires  pure (chunk tb /? ch)
  requires  pure (aligned 16 (core row))
  requires  pure (aligned_cont_layout (chunk tb) clrow)
  requires  pure (bounds_checked ==> k2 < n2)
  requires  pure (x %~ rx /\ vrow %~ rrow /\ vy %~ ry)
  requires  y |-> vy
  ensures exists* (vy' : lseq et n1).
    y |-> vy' ** pure (vy' %~ seq_fma' ch (rmapA (reveal rx)) #n2 (Seq.map_seq rmapB (chest1_to_seq rrow)) ry k1 k2)
{
  if (bounds_checked || k2 <^ n2) {
    let mut lchunk = [| zero #tb #_; ch |];
    assume pure (aligned 16 lchunk);
    chunk_end_bound n2 k2 ch;
    Math.divides_chain (chunk tb) ch k2;
    rewrite (row |-> Frac frow vrow)
         as (row |-> Frac frow (seq_to_chest1 (chest1_to_seq vrow)));
    Kuiper.Sparse.LoadStore.array_cpy_chunks ch lchunk row k2;
    rewrite (row |-> Frac frow (seq_to_chest1 (chest1_to_seq vrow)))
         as (row |-> Frac frow vrow);
    Kuiper.Chest1.Helpers.chest1_to_seq_approx vrow rrow;
    assert pure (Seq.slice (chest1_to_seq vrow) k2 (k2 + ch) %~
                 Seq.slice (chest1_to_seq rrow) k2 (k2 + ch));
    fma_arr mapA mapB rmapA rmapB x ch lchunk y
      #_ #rx #(Seq.slice (chest1_to_seq rrow) k2 (k2 + ch)) #ry k1;
  }

}
#pop-options

(* Keep the divisibility proof in a small pure context.  Reconstructing this
   refinement while elaborating the [SizeT] expression below is needlessly
   expensive once the scalar/sized dictionaries carry canonicality proofs. *)
#push-options "--z3rlimit 20"
let lemma_divides_vmprod_offset
  (ch : szp)
  (j k step : sz { fits (j + k * step * ch) })
  : Lemma
      (requires ch /? j)
      (ensures ch /? (j +^ k *^ step *^ ch))
  = lemma_divides_product ch (k * step);
    lemma_divides_sum ch j (k * step * ch)
#pop-options


#push-options "--z3rlimit 20"
let vmprod_chunk_bounds
  (n j step cols : nat) (ch : pos) (k : natlt (n / ch)) (checked : bool)
  : Lemma
      (requires fits n /\ fits (j + n * step) /\
                (checked ==> n == ch /\ j < cols))
      (ensures k * ch + ch <= n /\ fits (k * ch) /\
               fits (k * step) /\ fits (k * step * ch) /\
               fits (j + k * step * ch) /\
               (checked ==> j + k * step * ch < cols))
  = Spec.chunk_prefix_bound ch n (k + 1);
    FStar.Math.Lemmas.lemma_mult_le_right step (k * ch) n;
    FStar.Math.Lemmas.lemma_mult_le_left (k * step) 1 ch;
    if checked then assert (k == 0)

inline_for_extraction noextract
fn load_vmprod_row
  (#ta #tb #et : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb, sized tb, has_vec_cpy tb,
     scalar et, real_like et |}
  (mapA : ta -> et)
  (mapB : tb -> et)
  (rmapA : real -> real { MU.approx1 mapA rmapA })
  (rmapB : real -> real { MU.approx1 mapB rmapB })
  (ch : szp)
  (#n1 : sz { ch /? n1 })
  (y : larray et n1)
  (#vy : erased (lseq et n1))
  (x : ta)
  (#rx : erased real)
  (#ry : erased (lseq real n1))
  (#n2 : sz { ch /? n2 })
  (#lrow : layout1 n2) {| ctlayout lrow, clrow : cont_layout lrow |}
  (row : array1 tb lrow)
  (#frow : perm)
  (#vrow : chest1 tb n2)
  (#rrow : chest1 real n2)
  (j : sz { ch /? j })
  (step : sz)
  (bounds_checked : bool)
  preserves gpu
  preserves row |-> Frac frow vrow
  requires  pure (chunk tb /? ch)
  requires  pure (aligned 16 (core row))
  requires  pure (aligned_cont_layout (chunk tb) clrow)
  requires  pure (fits (j + n1 * step))
  requires  pure (bounds_checked ==> n1 == ch /\ j < n2)
  requires  pure (x %~ rx /\ vrow %~ rrow /\ vy %~ ry)
  requires  y |-> vy
  ensures exists* (vy' : lseq et n1).
    y |-> vy' ** pure (vy' %~ Spec.seq_load_vmprod_row ch #n1 ry (rmapA (reveal rx)) #n2 (Seq.map_seq rmapB (chest1_to_seq rrow)) j step (n1 / ch))
{
  let mut k : sz = 0sz;

  while (!k <^ n1 /^ ch)
    invariant exists* vk (vy' : lseq et n1).
      k |-> vk **
      y |-> vy' **
      pure (
        vk <= n1 / ch /\
        vy' %~ Spec.seq_load_vmprod_row ch #n1 ry (rmapA (reveal rx)) #n2 (Seq.map_seq rmapB (chest1_to_seq rrow)) j step vk
      )
    decreases (n1 /^ ch - !k)
  {
    with vk. assert k |-> vk;
    vmprod_chunk_bounds (v n1) (v j) (v step) (v n2) ch (v vk) bounds_checked;
    lemma_divides_vmprod_offset ch j vk step;
    load_vmprod_chunk mapA mapB rmapA rmapB ch
      y x #rx #(Spec.seq_load_vmprod_row ch #n1 ry (rmapA (reveal rx)) #n2 (Seq.map_seq rmapB (chest1_to_seq rrow)) j step !k)
      row #_ #_ #rrow
      (!k *^ ch) (j +^ !k *^ step *^ ch)
      bounds_checked;
    k := !k +^ 1sz;
  };
}

#pop-options

open Kuiper.Array2.Strided { strided_row_major, aligned_strided_row_major }

inline_for_extraction noextract
fn tile_mm_
  (#ta #tb #et : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb, sized tb, has_vec_cpy tb,
     scalar et, real_like et |}
  (mapA : ta -> et)
  (mapB : tb -> et)
  (rmapA : real -> real { MU.approx1 mapA rmapA })
  (rmapB : real -> real { MU.approx1 mapB rmapB })
  (ch : szp)
  (#m1 #n1 : sz { ch /? n1 })
  (y : larray et n1)
  (#vy : erased (lseq et n1))
  (#ry : erased (lseq real n1))
  (elems : larray ta m1)
  (row_ind : larray sz m1)
  (#fx : perm)
  (#m2 #n2 : szp { ch /? n2 })
  (#lm : layout2 m2 n2) {| ctlayout lm, srm : strided_row_major lm |}
  (m : array2 tb lm)
  (#fm : perm)
  (#em : chest2 tb m2 n2)
  (#rm : chest2 real m2 n2)
  (j : sz { ch /? j })
  (step : sz)
  (to : szle m1)
  (#velems : erased (lseq ta to))
  (#relems : erased (lseq real to))
  (#vrow_ind : erased (lseq sz to))
  (#_ : squash (in_bounds 0 m2 (cast_pos vrow_ind)))
  preserves gpu
  preserves pts_to_slice elems   #fx 0 to velems
  preserves pts_to_slice row_ind #fx 0 to vrow_ind
  preserves m |-> Frac fm em
  requires  pure (chunk tb /? ch)
  requires  pure (aligned 16 (core m) /\ aligned_strided_row_major (chunk tb) srm)
  requires  pure (fits (j + n1 * step))
  requires  pure (vy %~ ry /\ velems %~ relems /\ em %~ rm)
  requires  y |-> vy
  ensures exists* (vy' : lseq et n1).
    y |-> vy' ** pure (vy' %~ Spec.seq_load_vmprod ch #to #n1 ry (Seq.map_seq rmapA relems) (cast_pos vrow_ind) (Kuiper.Chest.chest_map rmapB rm) j step to)
{
  // All vector chunks owned by this thread start at or after j. Threads
  // outside the output can skip the reduction, but must still participate
  // in the caller's shared-memory loads and barriers.
  if (j <^ n2) {
    // With one vector chunk per thread, the outer check proves every load
    // in bounds. The static flag removes the check from the reduction loop.
    let mut k : sz = 0sz;

    while (!k <^ to)
      // invariant exists* vk (vy' : chest1 et n1).
      invariant exists* vk (vy' : lseq et n1).
        k |-> vk **
        y |-> vy' **
        pure (
          vk <= to /\
          vy' %~ Spec.seq_load_vmprod ch #to #n1 ry (Seq.map_seq rmapA relems) (cast_pos vrow_ind) (Kuiper.Chest.chest_map rmapB rm) j step vk
        )
      decreases (to - !k)
    {
      let kv = !k;
      let kr = slice_read row_ind kv;
      let kx = slice_read elems kv;

      // [kr] indexes a valid row of [m] by the sparsity bound.
      assert pure (v kr == cast_pos vrow_ind @! kv);

      tensor_extract_row_ro m (v kr);
      aligned_cont_strided_row_major lm (chunk tb) kr;

      load_vmprod_row mapA mapB rmapA rmapB ch
        y kx #(relems @! kv) #(Spec.seq_load_vmprod ch #to #n1 ry (Seq.map_seq rmapA relems) (cast_pos vrow_ind) (Kuiper.Chest.chest_map rmapB rm) j step kv)
        #_ #_ #(ctlayout_slice _ 0 (v kr)) // should not be needed
        (tensor_row m (v kr)) #_ #_ #(chest2_row rm (v kr)) j step (FStar.SizeT.eq n1 ch);

      tensor_restore_row m (v kr);

      k := !k +^ 1sz;
    }
  } else {
    Spec.seq_load_vmprod_outside ch #to #n1 ry (Seq.map_seq rmapA relems) (cast_pos vrow_ind) (Kuiper.Chest.chest_map rmapB rm) j step to;
    assert pure (Spec.seq_load_vmprod ch #to #n1 ry (Seq.map_seq rmapA relems) (cast_pos vrow_ind) (Kuiper.Chest.chest_map rmapB rm) j step to == reveal ry);
    assert pure (seq_approximates (reveal vy) (reveal ry));
  }
}

let intro_tile_mm_result
  (#et : Type0) {| scalar et, real_like et |}
  (rmapA rmapB : real -> real)
  (ch : pos)
  (#shared #cols : nat {  ch /? cols })
  (#ly : nat { ch /? ly })
  (y0 : erased (lseq real ly))
  (eB : chest2 real shared cols)
  (j : nat { ch /? j })
  (step : nat)
  (y : lseq et ly)
: Lemma (requires y %~ y0)
        (ensures tile_mm_result rmapA rmapB ch y0 (Seq.empty <: lseq real 0) Seq.empty eB j step y)
= ()

let mask_tile_mm_result
  (#et : Type0) {| scalar et, real_like et |}
  (rmapA rmapB : real -> real)
  (ch : pos)
  (#shared #cols : nat { ch /? cols })
  (#ly : nat { ch /? ly })
  (y0 : lseq real ly)
  (nnz : nat)
  (mask_len : natle nnz)
  (elems : erased (lseq real (nnz - mask_len)))
  (row_ind : erased (lseq nat nnz) { in_bounds 0 shared row_ind })
  (eB : chest2 real shared cols)
  (j : nat { ch /? j })
  (step : nat)
  (y : lseq et ly)
: Lemma
  (requires rmapA 0.0R == 0.0R /\
    tile_mm_result rmapA rmapB ch y0 (Seq.create mask_len zero @+ elems) row_ind eB j step y)
  (ensures  tile_mm_result rmapA rmapB ch y0 elems (Seq.slice row_ind mask_len nnz) eB j step y)
=
  map_zero_prefix rmapA mask_len elems;
  introduce forall (i : natlt ly).
    (let col = j + i / ch * step * ch + i % ch in
     col < cols ==> (y @! i) %~
       sparse_dprod_acc (y0 @! i) (Seq.map_seq rmapA elems) (Seq.slice row_ind mask_len nnz) (ematrix_col (Kuiper.Chest.chest_map rmapB eB) col))
  with (
    let col = j + i / ch * step * ch + i % ch in
    if col < cols then (
      _dprod_acc_mask_lemma (y0 @! i) mask_len (Seq.map_seq rmapA elems)
        (seq_make_sparse row_ind (ematrix_col (Kuiper.Chest.chest_map rmapB eB) col)) nnz;
      seq_make_sparse_slice row_ind mask_len nnz
        (ematrix_col (Kuiper.Chest.chest_map rmapB eB) col)
    )
  )

let tile_mm_result_lemma
  (#et : Type0) {| scalar et, real_like et |}
  (rmapA : real -> real { rmapA 0.0R == 0.0R })
  (rmapB : real -> real)
  (ch : pos)
  (#rows #shared #cols : nat { ch /? cols })
  (eA : chest2 real rows shared)
  (i : natlt rows)
  (#nnz : nat)
  (elems : erased (lseq real nnz))
  (row_ind : erased (lseq nat nnz))
  (#_ : squash (in_bounds 0 shared row_ind /\ sorted row_ind))
  (eB : chest2 real shared cols)
  (j : nat { ch /? j })
  (step : nat)
  (#ly : nat { ch /? ly })
  (y : lseq et ly)
: Lemma
  (requires
    seq_unsparse _ _ elems row_ind == ematrix_row eA i /\
    tile_mm_result rmapA rmapB ch
      (Seq.create ly zero)
      elems row_ind
      eB
      j step
      y
  )
  (ensures tile_values_approx ch (matmul (Kuiper.Chest.chest_map rmapA eA) (Kuiper.Chest.chest_map rmapB eB)) i j y step)
=
  introduce forall (ix : natlt ly).
    (let col = j + ix / ch * step * ch + ix % ch in
     col < cols ==> (y @! ix) %~ acc2 (matmul (Kuiper.Chest.chest_map rmapA eA) (Kuiper.Chest.chest_map rmapB eB)) i col)
  with (
    let col = j + ix / ch * step * ch + ix % ch in
    if col < cols then (
      map_unsparse rmapA elems row_ind shared;
      sparse_dprod_lemma (Seq.map_seq rmapA elems) row_ind
        (ematrix_col (Kuiper.Chest.chest_map rmapB eB) col);
      dprod_is_matmul_single (Kuiper.Chest.chest_map rmapA eA) (Kuiper.Chest.chest_map rmapB eB) i col
    )
  )

let tile_mm_witness
  (#et : Type0) {| scalar et, real_like et |}
  (rmapA rmapB : real -> real)
  (ch : pos)
  (#shared #cols : nat {  ch /? cols })
  (#ly : nat { ch /? ly })
  (y0 : erased (lseq real ly))
  (#lA : nat)
  (elems : erased (lseq real lA))
  (row_ind : erased (lseq nat lA) { in_bounds 0 shared row_ind })
  (eB : chest2 real shared cols)
  (j : nat { ch /? j })
  (step : nat)
  (y : lseq et ly)
  : Ghost (erased (lseq real ly))
      (requires tile_mm_result rmapA rmapB ch y0 elems row_ind eB j step y)
      (ensures fun ry -> y %~ ry /\
        Spec.tile_mm_result ch y0 (Seq.map_seq rmapA elems) row_ind
          (Kuiper.Chest.chest_map rmapB eB) j step ry)
  = hide (Seq.init_ghost ly (fun i ->
      let col = j + i / ch * step * ch + i % ch in
      if col < cols
      then sparse_dprod_acc (y0 @! i) (Seq.map_seq rmapA elems) row_ind (ematrix_col (Kuiper.Chest.chest_map rmapB eB) col)
      else to_real (y @! i)))

inline_for_extraction noextract
fn tile_mm
  (#ta #tb #et : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb, sized tb, has_vec_cpy tb,
     scalar et, real_like et |}
  (mapA : ta -> et)
  (mapB : tb -> et)
  (rmapA : real -> real { MU.approx1 mapA rmapA })
  (rmapB : real -> real { MU.approx1 mapB rmapB })
  (ch : szp)
  (#shared #cols : szp { ch /? cols } )
  (#ly : sz { ch /? ly })
  (y : larray et ly)
  (vy0 : erased (lseq real ly))
  (#vy : erased (lseq et ly))
  (#lA : sz) (elems : larray ta lA) (row_ind : larray sz lA) (#fA : perm)
  (#nnz : erased nat)
  (velems : erased (lseq ta nnz))
  (vrow_ind : erased (lseq sz nnz) { in_bounds 0 shared (cast_pos vrow_ind) })
  (relems : erased (lseq real nnz))
  (#lB : layout2 shared cols) {| ctlayout lB, srm : strided_row_major lB |}
  (mB : array2 tb lB) (#fB : perm) (#eB : chest2 tb shared cols)
  (#rB : chest2 real shared cols)
  (j : sz { ch /? j })
  (step : sz)
  (from to : erased nat { to <= nnz })
  (cant : szle lA { v cant == to - from })
  preserves gpu
  preserves pts_to_slice elems   #fA 0 cant (Seq.slice velems from to <: lseq ta cant)
  preserves pts_to_slice row_ind #fA 0 cant (Seq.slice vrow_ind from to <: lseq sz cant)
  preserves mB |-> Frac fB eB
  requires  pure (chunk tb /? ch)
  requires  pure (aligned 16 (core mB) /\ aligned_strided_row_major (chunk tb) srm)
  requires  pure (fits (j + ly * step))
  requires  pure (velems %~ relems /\ eB %~ rB)
  requires
    y |-> vy **
    pure (
      tile_mm_result rmapA rmapB ch
        vy0
        (Seq.slice relems 0 from <: lseq real from)
        (Seq.slice (cast_pos vrow_ind) 0 from)
        rB
        j step
        vy
    )
  ensures exists* (vy' : lseq et ly).
    y |-> vy' **
    pure (
      tile_mm_result rmapA rmapB ch
        vy0
        (Seq.slice relems 0 to <: lseq real to)
        (Seq.slice (cast_pos vrow_ind) 0 to <: lseq nat to)
        rB j step vy'
    )
{
  let ry = tile_mm_witness rmapA rmapB ch vy0 #from
    (Seq.slice relems 0 from) (Seq.slice (cast_pos vrow_ind) 0 from)
    rB j step vy;

  tile_mm_ mapA mapB rmapA rmapB ch y #vy #ry elems row_ind mB #fB #eB #rB j step cant
    #_ #(Seq.slice relems from to);
  Spec.seq_load_vmprod_step_lemma ch cant vy0 (Seq.map_seq rmapA relems)
    (cast_pos vrow_ind) from (v cant) (Kuiper.Chest.chest_map rmapB rB) j step ry;

  assert pure (Seq.equal
    (cast_pos #cant (Seq.slice vrow_ind from to))
    (Seq.slice (cast_pos vrow_ind) from to));
  with vy'. assert y |-> vy';
  assert pure (tile_mm_result rmapA rmapB ch vy0 #to (Seq.slice relems 0 to)
    (Seq.slice (cast_pos vrow_ind) 0 to) rB j step vy');
}
