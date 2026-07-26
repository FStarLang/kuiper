module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.TeardownHelpers

#lang-pulse

open Kuiper
#set-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

open Kuiper.EMatrix
open Kuiper.Tensor
open Kuiper.Tensor.Tiling

module SZ = Kuiper.SizeT

open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

ghost
fn join_lane_cells_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (m : array2 et l)
  (r : chest2 real rows cols)
  requires
    forall+ (lane : natlt warp_size).
      exists* (em : chest2 et rows cols).
        own_lane_cells m em lane ** pure (em %~ r)
  ensures
    exists* (em : chest2 et rows cols).
      m |-> em ** pure (em %~ r)

ghost
fn array2_untile_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (trows : pos{trows /? rows})
  (tcols : pos{tcols /? cols})
  {| enumerable (natlt (rows / trows)),
     enumerable (natlt (cols / tcols)) |}
  (r : chest2 real rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (#_ : squash (SZ.fits (rows / trows)))
  (#_ : squash (SZ.fits (cols / tcols)))
  requires
    forall+ (tr : natlt (rows / trows))
             (tc : natlt (cols / tcols)).
      exists* (em : chest2 et trows tcols).
        array2_subtile m trows tcols tr tc |-> em **
        pure (em %~ ematrix_subtile r trows tcols tr tc)
  ensures
    exists* (em : chest2 et rows cols).
      m |-> em ** pure (em %~ r)
