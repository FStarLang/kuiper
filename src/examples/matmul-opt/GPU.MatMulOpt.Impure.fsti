module GPU.MatMulOpt.Impure

#lang-pulse

#push-options "--fuel 1 --ifuel 1"

open FStar.Mul
open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open GPU

module SZ = FStar.SizeT
open FStar.SizeT

let gpu_pts_to_matrix #a (rows columns: nat) (ga : gpu_array a (rows * columns)) (shared: erased pos) (s: erased (Seq.seq a)): slprop =
  gpu_pts_to_array ga #(1.0R /. Real.of_int shared) s

val gpu_matrix_share_underspec
  (#a:Type u#0)
  (#uid: int)
  (rows columns: nat)
  (ga : gpu_array a (rows * columns))
  (shared: erased pos)
  (s: erased (Seq.seq a) { Seq.length s == rows * columns })
: stt_ghost
    unit
    emp_inames
    (gpu_pts_to_matrix #a rows columns ga 1 s)
    (fun _ -> bigstar #uid 0 shared (fun _ -> gpu_pts_to_matrix #a rows columns ga shared s))

val gpu_matrix_unshare_underspec
  (#a:Type u#0)
  (#uid: int)
  (rows columns: nat)
  (ga : gpu_array a (rows * columns))
  (shared: erased pos)
  (s: erased (Seq.seq a) { Seq.length s == rows * columns })
: stt_ghost
    unit
    emp_inames
    (bigstar #uid 0 shared (fun _ -> gpu_pts_to_matrix #a rows columns ga shared s))
    (fun _ -> gpu_pts_to_matrix #a rows columns ga 1 s)

fn gpu_matrix_read
  #a
  (#rows #columns: SZ.t)
  (ga : gpu_array a (rows * columns))
  (#shared: erased pos)
  (#s: erased (Seq.seq a) { Seq.length s == rows * columns })
  (row: SZ.t{SZ.v row < rows})
  (col: SZ.t{SZ.v col < columns})
  requires gpu ** gpu_pts_to_matrix rows columns ga shared s
  returns v: a
  // TODO: is the assert here opaque?
  ensures gpu ** gpu_pts_to_matrix rows columns ga shared s ** pure (assert ((SZ.v row + 1) * columns <= rows * columns); v == Seq.index s (row * columns + SZ.v col))
{
  assume_ (pure (forall (x:nat). SizeT.fits x)); // CHEATING overflow
  unfold gpu_pts_to_matrix rows columns ga shared s;
  unfold gpu_pts_to_array ga #(Real.one /. Real.of_int shared) s;
  // TODO: strange that commenting this out causes an error
  assert (pure ((row + 1) * columns <= rows * columns));
  let idx = row *^ columns +^ col;
  let v = gpu_array_read #a #(rows * columns) #0 #(rows * columns) ga #(Real.one /. Real.of_int shared) idx #s;
  fold gpu_pts_to_array ga #(Real.one /. Real.of_int shared) s;
  fold gpu_pts_to_matrix rows columns ga shared s;
  v
}
#pop-options
