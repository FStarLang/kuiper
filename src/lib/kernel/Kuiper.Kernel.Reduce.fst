module Kuiper.Kernel.Reduce

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Kernel.RowReduce { row_reduce }
open Kuiper.Bijection
module SZ = Kuiper.SizeT
module C = Kuiper.Matrix.Casts

#set-options "--split_queries always"

let cbij (lena : szp)
  : (conc (lena @| INil) ==~ conc (1 @| lena @| INil)) =
  mk_cbij
    #(conc (lena @| INil))
    #(conc (1 @| lena @| INil))
    (function (i, ()) -> (0sz, (i, ())))
    (function (_, (i, ())) -> (i, ()))
    ez
    ez

// Very VERY tedious to reuse the per-row kernel for a flat array.
// But possible.
// We should kill the remaining admits, and refactor cbij to avoid spurious diamonds.
inline_for_extraction noextract
fn reduce1
  (#et : Type0) {| scalar et, real_like et |}
  (pre_map : et -> et)
  (pre_map_r : real -> real { pre_map %~ pre_map_r })
  (len : szp)
  (nth : szp { nth <= max_threads /\ SZ.fits (len + nth) })
  (#l : layout1 len) {| ctlayout l |}
  (x  : array1 et l  { is_global x })
  (#sx   : chest1 et len)
  (vr    : chest1 real len)
  preserves
    cpu **
    on gpu_loc (x |-> sx)
  requires
    pure (sx %~ vr)
  returns
    res : et
  ensures
    pure (res %~ chest1_rsum (chest_map pre_map_r vr))
{
  let x' = relay x (C.l1_to_l2 l);
    assert rewrites_to x' (relay x (C.l1_to_l2 l));
  map_loc gpu_loc
    #(x |-> sx)
    #(x' |-> C.c1_to_c2 sx)
    fn _ {
      C.t1_to_t2 x;
    };
  let out0 = gpu_array_alloc #et 1sz;
  let out = from_array (l1_forward 1) out0;
    assert rewrites_to out (from_array (l1_forward 1) out0);
  with s. assert on gpu_loc (out0 |-> s);
  map_loc gpu_loc
    #(out0 |-> s)
    #(out |-> mk1 (fun _ -> Seq.index s 0))
    fn _ {
      tensor_abs' (l1_forward 1) out0;
      with s'. assert out |-> s';
      assert pure (equal s' (mk1 (fun _ -> Seq.index s 0)));
    };
  assume pure (C.l1_to_l2 l == C.layout_bij (C.bij_up (cbij len)) l);
  (* ^ FIXME, diamonds *)
  row_reduce #et pre_map pre_map_r 1sz len nth
    #_ #(C.clayout_bij (cbij _) _) #_ #_
    x' out (C.c1_to_c2 vr);

  map_loc gpu_loc
    #(x' |-> C.c1_to_c2 sx)
    #(x |-> sx)
    fn _ {
      C.t2_to_t1 x';
      assert relay (relay x (C.l1_to_l2 l)) (C.l2_to_l1 (C.l1_to_l2 l))
               |-> C.c2_to_c1 (C.c1_to_c2 sx);
      assert pure (equal (C.c2_to_c1 (C.c1_to_c2 sx)) sx);
      assert relay (relay x (C.l1_to_l2 l)) (C.l2_to_l1 (C.l1_to_l2 l))
               |-> sx;
      assume pure (C.l2_to_l1 (C.l1_to_l2 l) == l); // sigh, extensionality of layouts
      rewrite each
        relay (relay x (C.l1_to_l2 l)) (C.l2_to_l1 (C.l1_to_l2 l))
      as x;
      ()
    };

  with s'. assert on gpu_loc (out |-> s');
  map_loc gpu_loc
    #(out |-> s')
    #(out0 |-> to_seq (l1_forward 1) s')
    fn _ {
      tensor_concr out;
      rewrite each core out as out0;
    };

  let local_out = Pulse.Lib.Vec.alloc #et zero 1sz;

  gpu_memcpy_device_to_host local_out out0 1sz;

  let res = Pulse.Lib.Vec.op_Array_Access local_out 0sz;

  assert pure (res %~ chest1_rsum (chest_map pre_map_r (chest2_row (C.c1_to_c2 vr) 0)));
  assert pure (equal (chest2_row (C.c1_to_c2 vr) 0) vr);
  assert pure (res %~ chest1_rsum (chest_map pre_map_r vr));

  Pulse.Lib.Vec.free local_out;
  gpu_array_free out0;

  res
}
