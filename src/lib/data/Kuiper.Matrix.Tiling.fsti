module Kuiper.Matrix.Tiling
#lang-pulse

(* An assumed API for tiling matrices. This will be implemented
   with array views, eventually. *)

open Kuiper
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix

val ematrix_subtile
  (#et : _)
  (#rows #cols : erased nat)
  (em : ematrix et rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : ematrix et trows tcols

val subtile_layout
  (#rows #cols : erased nat)
  (l : mlayout rows cols)
  (trows : pos {trows /? rows})
  (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows))
  (tc : natlt (cols / tcols))
  : mlayout trows tcols

instance val c_subtile_layout
  (#rows #cols : erased nat)
  (l : mlayout rows cols) {| clayout l |}
  (trows : erased pos {trows /? rows})
  (tcols : erased pos {tcols /? cols})
  (tr    : erased (natlt (rows / trows)))
  (tc    : erased (natlt (cols / tcols)))
  {| c_trows : concrete_sz (hide (reveal trows)),
     c_tcols : concrete_sz (hide (reveal tcols)),
     c_tr    : concrete_sz (hide (reveal tr)),
     c_tc    : concrete_sz (hide (reveal tc)),
  |}
  : clayout (subtile_layout l trows tcols tr tc)

val gpu_matrix_subtile
  (#et : _)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : erased nat {trows > 0 /\ trows /? rows})
  (tcols : erased nat {tcols > 0 /\ tcols /? cols})
  (tr : erased (natlt (rows / trows)))
  (tc : erased (natlt (cols / tcols)))
  : Tot (gpu_matrix et (subtile_layout l trows tcols tr tc))

ghost
fn gpu_matrix_tile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : ematrix et rows cols)
  requires
    gm |-> em
  ensures
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        gpu_matrix_subtile gm trows tcols tr tc |-> ematrix_subtile em trows tcols tr tc

ghost
fn gpu_matrix_untile
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (#em : ematrix et rows cols)
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
        gpu_matrix_subtile gm trows tcols tr tc |-> ematrix_subtile em trows tcols tr tc
  ensures
    gm |-> em

ghost
fn gpu_matrix_untile0
  (#et:Type0)
  (#rows #cols : nat)
  (#l : mlayout rows cols)
  (gm : gpu_matrix et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  requires
    forall+
      (tr : natlt (rows / trows))
      (tc : natlt (cols / tcols)).
      (exists* em.
        gpu_matrix_subtile gm trows tcols tr tc |-> em)
  ensures
    exists* em. gm |-> em
