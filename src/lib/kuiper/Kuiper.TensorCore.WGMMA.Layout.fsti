module Kuiper.TensorCore.WGMMA.Layout

open Kuiper
open Kuiper.Tensor
open Kuiper.Injection
open FStar.SizeT { (/^), (%^), (+^), ( *^ ) }

(* PTX K-major, no-swizzle layouts, in BF16 elements. Each core matrix
   contains 8x8 elements. LBO = 128 bytes, SBO = 256 bytes.
   https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions
   These are packed layouts, not ordinary row/column-major matrices. *)
let a_index (row : natlt 64) (k : natlt 16) : natlt 1024 =
  row / 8 * 128 + k / 8 * 64 + row % 8 * 8 + k % 8

let b_index (k : natlt 16) (col : natlt 8) : natlt 128 =
  k / 8 * 64 + col * 8 + k % 8

let a_layout : layout2 64 16 = {
  ulen = 1024;
  imap = { f = (fun (row, (k, ())) -> a_index row k); };
}

let b_layout : layout2 16 8 = {
  ulen = 128;
  imap = { f = (fun (k, (col, ())) -> b_index k col); };
}

inline_for_extraction noextract
instance c_a_layout : ctlayout a_layout = {
  ulen_fits = ();
  all_fit = ();
  cimap = (fun (i : conc (64 @| 16 @| INil)) ->
    let row, (k, ()) = i in
    row /^ 8sz *^ 128sz +^ k /^ 8sz *^ 64sz +^ row %^ 8sz *^ 8sz +^ k %^ 8sz);
}

inline_for_extraction noextract
instance c_b_layout : ctlayout b_layout = {
  ulen_fits = ();
  all_fit = ();
  cimap = (fun (i : conc (16 @| 8 @| INil)) ->
    let k, (col, ()) = i in k /^ 8sz *^ 64sz +^ col *^ 8sz +^ k %^ 8sz);
}
