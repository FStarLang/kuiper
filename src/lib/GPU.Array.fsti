module GPU.Array

// let with_pure (p:prop) (f : squash p -> vprop) : vprop =
//   pure p ** f ()

open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open FStar.Seq
open GPU.Base
module A = Pulse.Lib.Array

val gpu_array (a:Type u#0) (sz:nat) : Type u#0

val gpu_pts_to_array_slice
  (#a:Type u#0)
  (#sz:nat)
  (x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (i:nat) (j:nat)
  (v : seq a)
: vprop

let gpu_pts_to_array
  (#a:Type u#0)
  (#sz:nat)
  (x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (v : seq a)
: vprop
=
  gpu_pts_to_array_slice x #f 0 sz v

val gpu_pts_to_slice_ref
  (#a:Type u#0)
  (#sz:nat)
  (x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (i:nat) (j:nat)
  (#v : seq a)
  : stt_ghost unit emp_inames
      (gpu_pts_to_array_slice x #f i j v)
      (fun _ -> gpu_pts_to_array_slice x #f i j v ** pure (i <= j /\ j <= sz /\ Seq.length v == (j-i)))

val gpu_array_alloc
  (#a:Type u#0)
  (sz:nat)
: stt (gpu_array a sz)
      cpu
      (fun x ->
        cpu **
        (exists* (s:seq a). gpu_pts_to_array x #1.0R s))

val gpu_array_free
  (#a:Type u#0)
  (#sz:nat)
  (r : gpu_array a sz)
  (#v : erased (seq a))
: stt unit
      (cpu ** gpu_pts_to_array r #1.0R v)
      (fun _ -> cpu)

val gpu_array_read
  (#a:Type u#0)
  (#sz:nat) (#i:nat) (#j : nat{i <= j /\ j <= sz})
  (r:gpu_array a sz)
  (#f:perm)
  (idx:nat{i <= idx /\ idx < j})
  (#s : erased (seq a))
: stt a
      (gpu ** gpu_pts_to_array_slice #a #sz r #f i j s)
      (fun x ->
        gpu **
        gpu_pts_to_array_slice #a #sz r #f i j s **
        pure (
          i <= j /\ j <= sz /\
          Seq.length s == (j-i) /\
          x == Seq.index s (idx - i)))

val gpu_array_write
  (#a:Type u#0)
  (#sz:nat) (#i:nat) (#j : nat{i <= j /\ j <= sz})
  (r:gpu_array a sz)
  (idx:nat{i <= idx /\ idx < j})
  (v : a)
  (#s : erased (seq a))
: stt unit
      (gpu ** gpu_pts_to_array_slice #a #sz r #1.0R i j s)
      (fun _ ->
        exists* (s' : seq a).
          gpu **
          gpu_pts_to_array_slice #a #sz r #1.0R i j s' **
          pure (
            i <= j /\ j <= sz /\
            Seq.length s == (j-i) /\
            s' == Seq.upd s (idx - i) v))

val gpu_memcpy_host_to_device
  (#a:Type u#0)
  (#sz : nat)
  (arr : array a)
  (#f : perm)
  (#v : erased (seq a))
  (garr : gpu_array a sz)
  (#gv : erased (seq a))
: stt unit
      (cpu ** A.pts_to arr #f v ** gpu_pts_to_array garr #1.0R gv)
      (fun _ -> cpu ** A.pts_to arr #f v ** gpu_pts_to_array garr #1.0R v)

val gpu_memcpy_device_to_host
  (#a:Type u#0)
  (#sz : nat)
  (arr : array a)
  (#v : erased (seq a))
  (garr : gpu_array a sz)
  (#f : perm)
  (#gv : erased (seq a))
: stt unit
      (cpu ** A.pts_to arr #1.0R v ** gpu_pts_to_array garr #f gv)
      (fun _ -> cpu ** A.pts_to arr #1.0R gv ** gpu_pts_to_array garr #f gv ** pure (Seq.length gv == sz))

let gpu_pts_to_array1
  (#a:Type0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  (i:nat)
: vprop =
  exists* s. gpu_pts_to_array_slice arr i (i+1) s

val gpu_array_slice_1
  (#a:Type u#0)
  (#[exact (`0)] uid: int) (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a) { Seq.length v == sz })
: stt_ghost
    unit
    emp_inames
    (gpu_pts_to_array arr #f v)
    (fun _ -> bigstar #uid 0 sz (fun i -> gpu_pts_to_array_slice arr #f i (i+1) (Seq.Base.cons (Seq.Base.index v i) Seq.Base.empty)))

val gpu_array_slice_1_underspec
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (#v : erased (seq a))
: stt_ghost
    unit
    emp_inames
    (gpu_pts_to_array arr #f v)
    (fun _ -> bigstar 0 sz (gpu_pts_to_array1 arr #f))

val gpu_array_unslice_1_underspec
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
: stt_ghost
    unit
    emp_inames
    (bigstar 0 sz (gpu_pts_to_array1 arr #f))
    (fun _ -> exists* v. gpu_pts_to_array arr #f v)

// ```xx
// fn memcpy_host_to_device
//   (arr : array a)
//   (#f : perm)
//   (#v : erased (seq a))
//   (garr : gpu_array a sz)
//   (#gv : erased (seq a))
//   requires cpu ** arr |-> #f v ** garr |-> #1.0R gv
//   ensures  cpu ** arr |-> #f v ** garr |-> #1.0R v

// fn memcpy_device_to_host
//   (arr : array a)
//   (#f : perm)
//   (#v : erased (seq a))
//   (garr : gpu_array a sz)
//   (#gv : erased (seq a))
//   requires cpu ** arr |-> #1.0R v  ** garr |-> #f gv
//   ensures  cpu ** arr |-> #1.0R gv ** garr |-> #f gv


// fn gpu_array_alloc
//   (#a:Type u#0)
//   (sz:nat)
//   requires cpu
//   returns  x : gpu_array a sz
//   ensures  exists* (s:seq a). cpu ** x |-> #1.0R s




