module Kuiper.TensorCore.MMA.Views

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.TensorCore.MMA

module RO = Kuiper.TensorRO
module SZ = Kuiper.SizeT
module Cast = Kuiper.Float.Casts.Base

let affine (#m #n:nat) (l:RO.vlayout2 m n) (offset row_stride col_stride:nat) =
  forall (i:natlt m) (j:natlt n).
    l.RO.imap (idx2 i j) == offset + i * row_stride + j * col_stride

let pad_a (#m:nat{m <= 16}) (#k:nat{k <= 16}) (a:chest2 bf16 m k)
  : chest2 bf16 16 16 =
  mk2 #bf16 #16 #16 (fun i j -> if i < m && j < k then acc2 a i j else zero #bf16)

let pad_b (#k:nat{k <= 16}) (#n:nat{n <= 8}) (b:chest2 bf16 k n)
  : chest2 bf16 16 8 =
  mk2 #bf16 #16 #8 (fun i j -> if i < k && j < n then acc2 b i j else zero #bf16)

let selected (mask:nat) (i:natlt 16) (j:natlt 8) : bool =
  let bit:pos = if i < 8 then (if j % 2 = 0 then 1 else 2)
                else (if j % 2 = 0 then 4 else 8) in
  (mask / bit) % 2 = 1

let filled (mask:nat) (v:chest2 float 16 8) (x:float) : chest2 float 16 8 =
  mk2 #float #16 #8 (fun i j -> if selected mask i j then x else acc2 v i j)

let added (mask:nat) (d s:chest2 float 16 8) : chest2 float 16 8 =
  mk2 #float #16 #8 (fun i j -> if selected mask i j
    then Kuiper.Scalars.Base.add #float (acc2 d i j) (acc2 s i j)
    else acc2 d i j)

let rounded_window (row:nat{row <= 16}) (col:nat{col <= 8})
  (m:nat{row + m <= 16}) (n:nat{col + n <= 8}) (v:chest2 float 16 8)
  : chest2 bf16 m n =
  mk2 #bf16 #m #n (fun i j -> Cast.cast_f32_to_bf16 (acc2 v (row + i) (col + j)))

(* Native correspondence extensions of the existing MMA.fragment/emma.
   All 32 live lanes must execute matching calls convergently and publish
   input writes, just as for MMA.mma_sync. The gpu capability does not prove
   these collective obligations. No allocation, launch or synchronization
   is introduced, and the existing native numerical model is unchanged.

   Operand maps are read-only: zero strides may alias logical cells.
   Only in-bounds cells are loaded; absent rows/columns are register zeros.
   Owning output layouts retain their existing injectivity requirement.
   A mask bit selects one of each lane's four native accumulator registers. *)

fn mma_sync
  (m:sz{SZ.v m <= 16}) (k:sz{SZ.v k <= 16}) (n:sz{SZ.v n <= 8})
  (#la:RO.vlayout2 (SZ.v m) (SZ.v k))
  (a:RO.roarray2 bf16 la) (a_offset a_row_stride a_col_stride:sz)
  (#lb:RO.vlayout2 (SZ.v k) (SZ.v n))
  (b:RO.roarray2 bf16 lb) (b_offset b_row_stride b_col_stride:sz)
  (fr:fragment)
  (#fa #fb:perm)
  (#va:chest2 bf16 (SZ.v m) (SZ.v k))
  (#vb:chest2 bf16 (SZ.v k) (SZ.v n))
  (#vc:chest2 float 16 8)
  (layout_ok:squash (
    affine la (SZ.v a_offset) (SZ.v a_row_stride) (SZ.v a_col_stride) /\
    affine lb (SZ.v b_offset) (SZ.v b_row_stride) (SZ.v b_col_stride)))
  preserves gpu
  preserves a |-> Frac fa va
  preserves b |-> Frac fb vb
  requires fr |-> vc
  ensures fr |-> emma vc (pad_a va) (pad_b vb)

fn fill_masked (fr:fragment) (mask:sz{SZ.v mask < 16}) (x:float)
  (#v:chest2 float 16 8)
  preserves gpu
  requires fr |-> v
  ensures fr |-> filled (SZ.v mask) v x

fn add_masked (dst src:fragment) (mask:sz{SZ.v mask < 16})
  (#vd #vs:chest2 float 16 8)
  preserves gpu
  preserves src |-> vs
  requires dst |-> vd
  ensures dst |-> added (SZ.v mask) vd vs

fn store_bf16
  (fr:fragment)
  (row:sz{SZ.v row <= 16}) (col:sz{SZ.v col <= 8})
  (m:sz{SZ.v row + SZ.v m <= 16})
  (n:sz{SZ.v col + SZ.v n <= 8})
  (#l:layout2 (SZ.v m) (SZ.v n)) {| strided_col_major l |}
  (c:array2 bf16 l)
  (#v:chest2 float 16 8)
  (#vc:chest2 bf16 (SZ.v m) (SZ.v n))
  preserves gpu
  preserves fr |-> v
  requires c |-> Frac (1.0R /. warp_size) vc
  ensures c |-> Frac (1.0R /. warp_size)
    (rounded_window (SZ.v row) (SZ.v col) (SZ.v m) (SZ.v n) v)

inline_for_extraction let () = ()
