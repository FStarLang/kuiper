module Kuiper.MatMul.Impure

#lang-pulse

open FStar.Mul
open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open Kuiper

open FStar.SizeT

ghost
fn gpu_matrix_share_underspec
  (#a:Type u#0)
  (#uid: int)
  (rows columns: nat)
  (ga : gpu_array a (rows * columns))
  (shared: erased nat{shared > 0})
  (s: erased (seq a) { len s == rows * columns })
  requires gpu_pts_to_matrix #a rows columns ga 1 s
  ensures bigstar #uid 0 shared (fun _ -> gpu_pts_to_matrix #a rows columns ga shared s)
{
  admit(); (* TODO: this is just splitting permissions and should have a model. *)
}

ghost
fn gpu_matrix_unshare_underspec
  (#a:Type u#0)
  (#uid: int)
  (rows columns: nat)
  (ga : gpu_array a (rows * columns))
  (shared: erased nat{shared > 0})
  (s: erased (seq a) { len s == rows * columns })
  requires bigstar #uid 0 shared (fun _ -> gpu_pts_to_matrix #a rows columns ga shared s)
  ensures  gpu_pts_to_matrix #a rows columns ga 1 s
{
  admit(); (* TODO: this is just splitting permissions and should have a model. *)
}
