module Kuiper.Example.MemcpySize

#lang-pulse

open Kuiper
open Pulse.Lib.Vec
module SZ = Kuiper.SizeT
module Seq = FStar.Seq

(* Keep the element count dynamic so the tests exercise the extracted
   multiplication at and above the 4 GiB byte-count boundary. *)
let host_to_device (#sz:erased nat) (dst:larray u64 sz) (src:vec u64)
  (cnt:SZ.t) (#f:perm) (#v #gv:erased (seq u64)) =
  Kuiper.Array.memcpy_host_to_device dst src cnt #f #v #gv

let device_to_host (#sz:erased nat) (dst:vec u64) (src:larray u64 sz)
  (cnt:SZ.t) (#f:perm) (#v #gv:erased (seq u64)) =
  Kuiper.Array.memcpy_device_to_host dst src cnt #f #v #gv

let device_to_device (#sz:erased nat) (dst src:larray u64 sz)
  (cnt:SZ.t) (#f:perm) (#v #gv:erased (seq u64)) =
  Kuiper.Array.memcpy_device_to_device dst src cnt #f #v #gv

let host_to_device_offset (#dst_sz:erased nat) (dst:larray u64 dst_sz)
  (dst_off:SZ.t) (#src_sz:erased nat) (src:vec u64) (src_off:SZ.t)
  (cnt:SZ.t{dst_off + cnt <= dst_sz /\ src_off + cnt <= src_sz})
  (#f:perm) (#v:erased (seq u64){Seq.length v == src_sz})
  (#gv:erased (seq u64){Seq.length gv == dst_sz}) =
  Kuiper.Array.memcpy_host_to_device' dst dst_off #src_sz src src_off cnt #f #v #gv

let device_to_host_offset (#dst_sz:erased nat) (dst:vec u64)
  (dst_off:SZ.t) (#src_sz:erased nat) (src:larray u64 src_sz) (src_off:SZ.t)
  (cnt:SZ.t{dst_off + cnt <= dst_sz /\ src_off + cnt <= src_sz})
  (#f:perm) (#v:erased (seq u64){Seq.length v == src_sz})
  (#gv:erased (seq u64){Seq.length gv == dst_sz}) =
  Kuiper.Array.memcpy_device_to_host' #_ #_ #dst_sz dst dst_off src src_off cnt #f #v #gv
