module Kuiper.Sparse.Mask

#lang-pulse

open Kuiper
open Kuiper.Sparse

inline_for_extraction noextract
fn mask_array
  (#et : Type0)
  (#n : sz)
  (x : larray et n)
  (to : szle n)
  (z : et)
  (nthr : sz)
  (tid : szlt nthr)
  preserves gpu
  requires thread_slice_live x 0 to nthr tid
  requires pure (fits (n + nthr))
  ensures thread_slice_pts_to_value x 0 to z nthr tid