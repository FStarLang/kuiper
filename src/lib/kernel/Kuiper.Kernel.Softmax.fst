module Kuiper.Kernel.Softmax

(* 1-D softmax, implemented by reusing the per-row 2-D kernel
   [Kuiper.Kernel.RowSoftmax.row_softmax_gpu]: a length-[lena] array is a
   [1 x lena] matrix, so its softmax is the (single) row-softmax of that matrix.
   This mirrors how [Kuiper.Kernel.Reduce.reduce1] reuses [row_reduce]. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Spec.Softmax
open Kuiper.Kernel.RowSoftmax { row_softmax_real, row_softmax_gpu }
module C = Kuiper.Matrix.Casts

#set-options ""

(* Spec bridge: the row-softmax of the [1 x lena] embedding of [ra], cast back
   to 1-D, is exactly the 1-D [softmax_real ra]. *)
let softmax_via_row_spec (#lena : nat) (ra : chest1 real lena)
  : Lemma (C.c2_to_c1 (row_softmax_real (C.c1_to_c2 ra)) == softmax_real ra)
  = ext (chest2_row (C.c1_to_c2 ra) 0) ra;
    ext (C.c2_to_c1 (row_softmax_real (C.c1_to_c2 ra))) (softmax_real ra);
    ()

inline_for_extraction noextract
fn softmax_gpu
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (nth : szp{nth <= max_threads})
  (#lena : szp)
  (#l : layout1 lena) {| ctlayout l |}
  (a : array1 et l { is_global a })
  (#va : chest1 et lena)
  (ra  : chest1 real lena)
  preserves
    cpu
  requires
    on gpu_loc (a |-> va) **
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
  ensures
    exists* (va' : chest1 et lena).
      on gpu_loc (a |-> va') **
      pure (va' %~ softmax_real ra)
{
  (* View the flat array as a [1 x lena] matrix. *)
  let a' = relay a (C.l1_to_l2 l);
    assert rewrites_to a' (relay a (C.l1_to_l2 l));
  map_loc gpu_loc
    #(a |-> va)
    #(a' |-> C.c1_to_c2 va)
    fn _ {
      C.t1_to_t2 a;
    };
  (* Run the per-row kernel on the single row. *)
  row_softmax_gpu #et 1sz lena nth
    #_ #(C.cl1_to_cl2 ()) a' (C.c1_to_c2 ra);
  with sa'. assert on gpu_loc (a' |-> sa');

  (* Cast the result back to 1-D. *)
  map_loc gpu_loc
    #(a' |-> sa')
    #(a |-> C.c2_to_c1 sa')
    fn _ {
      C.t2_to_t1_restore a;
    };

  softmax_via_row_spec ra;
  ()
}
