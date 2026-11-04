module Kuiper.Sparse.SPMM

#lang-pulse

open Kuiper
module MU = Kuiper.Kernel.GEMM.Util
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
open Kuiper.Tensor
open Kuiper.Sparse
open Kuiper.Sparse.Approx
open Kuiper.Sparse.SPMM.Defs { size_req, working_chunk }
open Kuiper.EMatrix
open Kuiper.Array.Vectorized
open Kuiper.Array2.Strided { strided_row_major, aligned_strided_row_major }

open Pulse.Lib.Pledge
open Kuiper.Kernel.Base


(* [eA] and [eB] describe the actual inputs; [rA] and [rB] are explicit real
   models. The CSR model additionally requires that every unstored position
   in [rA] is exactly real zero (see [csr_matrix_approx]). The contracts
   preserve the input descriptions and specify approximation of a real-valued
   result, without specifying exact computed floats. *)

(* The general entry points separate the sparse input, dense input, output,
   and accumulator types. [mapA] and [mapB] map loaded values before
   multiplication; [comb] combines the initial C value with the completed sum.
   [rmapA], [rmapB], and [rcomb] specify those operations over the reals.
   The sparse map must preserve real zero, including implicit zeros and padding.
   Sparse tiles use the input vector width. Column groups use
   [working_chunk tb tc]; each operand keeps its own stride alignment.
   Partial final tiles remain supported; blockItemsX need not divide cols.
   The result approximates [MS.gmmcomb rmapA rmapB rcomb rC rA rB].
   The single-type [spmm_on]/[spmm] wrappers use identity input maps and replace C. *)

(* Asynchronous, stream-parametric SpMM.

Launches on [s] and hands back a pledge redeemable once the stream's epoch
completes, rather than synchronizing internally. A caller that issues many
SpMMs can therefore create one stream, launch into it repeatedly, and
synchronize once, instead of paying a stream create/synchronize/destroy per
call as the synchronous [spmm] below does.

Note the resources that [spmm] merely [preserves] appear here in [requires] and
again under the pledge: the GPU borrows them for the duration of the launch, so
they are not available to the caller until the epoch is done. *)
inline_for_extraction noextract
fn gspmm_on
  (#et #tb #tc #tacc : Type0)
  {| scalar et, sized et, has_vec_cpy et, real_like et,
     scalar tb, sized tb, has_vec_cpy tb, real_like tb,
     scalar tc, sized tc, has_vec_cpy tc, real_like tc,
     scalar tacc, real_like tacc |}
  (mapA : et -> tacc)
  (mapB : tb -> tacc)
  (rmapA : real -> real { MU.approx1 mapA rmapA /\ rmapA 0.0R == 0.0R })
  (rmapB : real -> real { MU.approx1 mapB rmapB })
  (comb : tc -> tacc -> tc)
  (rcomb : real -> real -> real { approx2 comb rcomb })
  (rows shared cols : szp { working_chunk tb tc /? cols })
  (blockItemsK blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * working_chunk tb tc) /? blockItemsX
  }))
  (blockChunks : sz{
    SZ.v blockChunks == SZ.v blockItemsX / SZ.v blockWidth
  }) // Ver nota abajo
  (#lB : layout2 shared cols) {| ctlayout lB, srmB : strided_row_major lB |}
  (#lC : layout2 rows cols)   {| ctlayout lC, srmC : strided_row_major lC |}
  (gA : csr_matrix et (SZ.v rows) (SZ.v shared){is_global gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 tb lB{is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned_strided_row_major (chunk tb) srmB))
  (#fB : perm)
  (gC : array2 tc lC{is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned_strided_row_major (chunk tc) srmC))
  // matriz sparse gA
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 tb shared cols)
  (#rB : chest2 real shared cols)
  (#rC : chest2 real rows cols)
  (s : stream_t)
  (#e : epoch_t)
  norewrite
  preserves cpu ** stream_live s
  requires
    epoch_live s e **
    on gpu_loc (csr_matrix_approx gA #fA eA rA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (matrix_approx gC rC) **
    pure (eB %~ rB) **
    pure (size_req rows shared cols blockWidth blockItemsK blockItemsX)
  ensures
    epoch_live s (epoch_next e) **
    pledge0 (epoch_done s (epoch_next e)) (on gpu_loc (
      csr_matrix_approx gA #fA eA rA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      matrix_approx gC (MS.gmmcomb rmapA rmapB rcomb rC rA rB)
    ))

(* Synchronous SpMM: [gspmm_on] on a private stream, waited on before returning. *)
inline_for_extraction noextract
fn gspmm
  (#et #tb #tc #tacc : Type0)
  {| scalar et, sized et, has_vec_cpy et, real_like et,
     scalar tb, sized tb, has_vec_cpy tb, real_like tb,
     scalar tc, sized tc, has_vec_cpy tc, real_like tc,
     scalar tacc, real_like tacc |}
  (mapA : et -> tacc)
  (mapB : tb -> tacc)
  (rmapA : real -> real { MU.approx1 mapA rmapA /\ rmapA 0.0R == 0.0R })
  (rmapB : real -> real { MU.approx1 mapB rmapB })
  (comb : tc -> tacc -> tc)
  (rcomb : real -> real -> real { approx2 comb rcomb })
  (rows shared cols : szp { working_chunk tb tc /? cols })
  (blockItemsK blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * working_chunk tb tc) /? blockItemsX
  }))
  (blockChunks : sz{
    SZ.v blockChunks == SZ.v blockItemsX / SZ.v blockWidth
  }) // Ver nota abajo
  (#lB : layout2 shared cols) {| ctlayout lB, srmB : strided_row_major lB |}
  (#lC : layout2 rows cols)   {| ctlayout lC, srmC : strided_row_major lC |}
  (gA : csr_matrix et (SZ.v rows) (SZ.v shared){is_global gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 tb lB{is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned_strided_row_major (chunk tb) srmB))
  (#fB : perm)
  (gC : array2 tc lC{is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned_strided_row_major (chunk tc) srmC))
  // matriz sparse gA
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 tb shared cols)
  (#rB : chest2 real shared cols)
  (#rC : chest2 real rows cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (csr_matrix_approx gA #fA eA rA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    on gpu_loc (matrix_approx gC rC) **
    pure (eB %~ rB) **
    pure (size_req rows shared cols blockWidth blockItemsK blockItemsX)
  ensures on gpu_loc (matrix_approx gC (MS.gmmcomb rmapA rmapB rcomb rC rA rB))

inline_for_extraction noextract
fn spmm_on
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (rows shared cols : szp { chunk et /? cols })
  (blockItemsK blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * chunk et) /? blockItemsX
  }))
  (blockChunks : sz{
    SZ.v blockChunks == SZ.v blockItemsX / SZ.v blockWidth
  }) // Ver nota abajo
  (#lB : layout2 shared cols) {| ctlayout lB, srmB : strided_row_major lB |}
  (#lC : layout2 rows cols)   {| ctlayout lC, srmC : strided_row_major lC |}
  (gA : csr_matrix et (SZ.v rows) (SZ.v shared){is_global gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 et lB{is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmB))
  (#fB : perm)
  (gC : array2 et lC{is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmC))
  // matriz sparse gA
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 et shared cols)
  (#rB : chest2 real shared cols)
  (#eC : chest2 et rows cols)
  (s : stream_t)
  (#e : epoch_t)
  norewrite
  preserves cpu ** stream_live s
  requires
    epoch_live s e **
    on gpu_loc (csr_matrix_approx gA #fA eA rA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (live gC) **
    pure (eB %~ rB) **
    pure (size_req rows shared cols blockWidth blockItemsK blockItemsX)
  ensures
    epoch_live s (epoch_next e) **
    pledge0 (epoch_done s (epoch_next e)) (on gpu_loc (
      csr_matrix_approx gA #fA eA rA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      matrix_approx gC (MS.matmul rA rB)
    ))

inline_for_extraction noextract
fn spmm
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (rows shared cols : szp { chunk et /? cols })
  (blockItemsK blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * chunk et) /? blockItemsX
  }))
  (blockChunks : sz{
    SZ.v blockChunks == SZ.v blockItemsX / SZ.v blockWidth
  }) // Ver nota abajo
  (#lB : layout2 shared cols) {| ctlayout lB, srmB : strided_row_major lB |}
  (#lC : layout2 rows cols)   {| ctlayout lC, srmC : strided_row_major lC |}
  (gA : csr_matrix et (SZ.v rows) (SZ.v shared){is_global gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 et lB{is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmB))
  (#fB : perm)
  (gC : array2 et lC{is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmC))
  // matriz sparse gA
  (#eA : chest2 et rows shared)
  (#rA : chest2 real rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 et shared cols)
  (#rB : chest2 real shared cols)
  (#eC : chest2 et rows cols)
  norewrite
  preserves
    cpu **
    on gpu_loc (csr_matrix_approx gA #fA eA rA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    on gpu_loc (live gC) **
    pure (eB %~ rB) **
    pure (size_req rows shared cols blockWidth blockItemsK blockItemsX)
  ensures on gpu_loc (matrix_approx gC (MS.matmul rA rB))
