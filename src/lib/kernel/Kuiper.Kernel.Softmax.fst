module Kuiper.Kernel.Softmax

(* 1-D softmax, implemented by reusing the per-row 2-D kernel
   [Kuiper.Kernel.RowSoftmax.row_softmax_gpu]: a length-[lena] array is a
   [1 x lena] matrix, so its softmax is the (single) row-softmax of that matrix.
   This mirrors how [Kuiper.Kernel.Reduce.reduce1] reuses [row_reduce]. *)

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Spec.Softmax
open Kuiper.Kernel.RowSoftmax { row_softmax_real, row_softmax_gpu }
open Kuiper.Bijection
module SZ = Kuiper.SizeT
module C = Kuiper.Matrix.Casts

#set-options "--split_queries always"

(* The 1<->2 index bijection used to view a flat array as a 1-row matrix
   (same as in [Kuiper.Kernel.Reduce]). *)
inline_for_extraction noextract
let cbij (lena : szp)
  : (conc (lena @| INil) ==~ conc (1 @| lena @| INil)) =
  mk_cbij
    #(conc (lena @| INil))
    #(conc (1 @| lena @| INil))
    (function (i, ()) -> (0sz, (i, ())))
    (function (_, (i, ())) -> (i, ()))
    ez
    ez

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
  assume pure (C.l1_to_l2 l == C.layout_bij (C.bij_up (cbij lena)) l);
  (* ^ FIXME, diamonds (as in Kuiper.Kernel.Reduce) *)

  (* Run the per-row kernel on the single row. *)
  row_softmax_gpu #et 1sz lena nth
    #_ #(C.clayout_bij (cbij _) _) a' (C.c1_to_c2 ra);
  with sa'. assert on gpu_loc (a' |-> sa');

  (* Cast the result back to 1-D. *)
  map_loc gpu_loc
    #(a' |-> sa')
    #(a |-> C.c2_to_c1 sa')
    fn _ {
      C.t2_to_t1 a';
      assert relay (relay a (C.l1_to_l2 l)) (C.l2_to_l1 (C.l1_to_l2 l))
               |-> C.c2_to_c1 sa';
      assume pure (C.l2_to_l1 (C.l1_to_l2 l) == l); // sigh, extensionality of layouts
      rewrite each
        relay (relay a (C.l1_to_l2 l)) (C.l2_to_l1 (C.l1_to_l2 l))
      as a;
      ()
    };

  softmax_via_row_spec ra;
  ()
}
