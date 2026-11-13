module Kuiper.Tensor.Runtime
#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Send

module A = Pulse.Lib.Array
module KA = Kuiper.Array.Core
module KL = Kuiper.Locs
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module Pledge = Pulse.Lib.Pledge
module S = FStar.Seq.Base
module BF16 = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64

inline_for_extraction noextract
let _ = ()

(* EXTERNAL TRUST, not an allocator implementation or a kernel postcondition.
   This interface describes the pinned native, enabled caching allocator on
   cuda:0, with a stable calling-thread current stream, default pool, no graph
   capture, expandable segments, custom allocator, OOM policy or user callback.
   Deterministic empty filling is inactive (the native empty_cuda wrapper can
   otherwise enqueue fill work); dispatch/profiler hooks must not add work.
   These are conformance assumptions for this boundary, NOT historical facts.

   runtime owns the pool invariant, including pending retired byte ranges,
   splitting/coalescing, locking and safe reclamation. "preserves runtime"
   restores this predicate; it does not leave its internal state unchanged.
   Other threads may use the allocator only through that invariant. All uses
   of client-owned storage are on stream; its unique epoch_live accounts for
   ALL earlier work, including previous invocations. No fresh token is minted.

   empty and final release enqueue no source work. They leave the queue tail
   unchanged, consistently with Kuiper.Epoch. The allocation may reuse bytes
   still in use before that tail: only a pledge is returned. Reclamation that
   the native allocator performs internally cannot yield epoch_done here. *)
val runtime (stream:Stream.stream_t) : slprop

val tensor (a:Type0) : Type0

(* Logical projections, not native accessor implementations. A tensor names a
   complete requested storage generation, not a mutable native Block pointer.
   Rounded padding remains runtime-owned. Equal reused addresses do not mint
   the old generation's owning capability. *)
val data (#a:Type0) (value:tensor a) : GTot (A.array a)
val dimensions (#a:Type0) (value:tensor a) : GTot (list pos)

(* Exclusive authority for the final owning reference. Copying the pure tensor
   name does not copy this capability. No conversion from a borrowed Tensor,
   subarray, raw pointer or storage pledge is provided. Retaining it across
   calls retains the native TensorImpl/StorageImpl/DataPtr chain. An owning
   result transfer moves this authority to the result; local destruction of
   a nonfinal reference is NOT release. There is no resize/alias-export API. *)
val owner (#a:Type0) (stream:Stream.stream_t) (value:tensor a) : slprop

noeq type dtype : Type0 -> Type0 =
  | BFloat16 : dtype BF16.t
  | Float32 : dtype F32.t
  | Int32 : dtype I32.t
  | Int64 : dtype I64.t

inline_for_extraction noextract
let rec numel (shape:list pos) : Tot pos =
  match shape with
  | [] -> 1
  | dim::rest -> dim * numel rest

inline_for_extraction noextract
let metadata (#a:Type0) (value:tensor a) (shape:list pos) : prop =
  dimensions value == shape /\
  A.length (data value) == numel shape /\
  KA.is_global_array (data value)

inline_for_extraction noextract
let contents (#a:Type0) (value:tensor a) : slprop =
  exists* (values:S.seq a). on KL.gpu_loc (A.pts_to (data value) values)

val allocation_error : Type0

noeq type allocation_result (a:Type0) =
  | Failed : error:allocation_error -> allocation_result a
  | Allocated : value:tensor a -> allocation_result a

(* Outcome abstraction of torch::empty, not an extracted exception-catching
   shim. Allocated describes ONLY a successful native normal return with the
   requested positive contiguous shape, dtype and cuda:0 device. Native errors
   (including OOM) propagate as Failed, retaining error identity; they provide
   no tensor or success theorem. Source refinement of exceptional unwinding
   remains a caller obligation. No success/existence premise is manufactured.
   Neither branch promises numerical values or initialized outputs. *)
noextract
fn empty
  (#a:Type0) (kind:dtype a)
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (shape:list pos{numel shape <= 2147483647})
  (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** runtime stream **
    Epoch.epoch_live stream epoch
  requires Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
  returns outcome:allocation_result a
  ensures (
    match outcome with
    | Failed _ -> Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
    | Allocated value ->
      owner stream value ** pure (metadata value shape) **
      Pledge.pledge0 (Epoch.epoch_done stream epoch) (contents value ** frame))

(* The actual native typed data_ptr accessor, with a live owning reference.
   Its result names the requested tensor extent, potentially a subrange of a
   larger cached allocation. It grants no is_full_array or raw-free authority.
   Reading metadata/pointers is not reading GPU cells and yields no ownership
   of their contents. The dtype index excludes native type-mismatch errors. *)
noextract
fn data_ptr
  (#a:Type0) (stream:Stream.stream_t) (value:tensor a)
  preserves KL.cpu ** owner stream value
  returns address:A.array a
  ensures pure (address == data value)

(* The real final owning-reference release through TensorImpl -> StorageImpl
   -> DataPtr's native deleter, NOT raw cudaFree and NOT a ghost body.
   Exact full storage plus its pending dependency enters the runtime-owned
   invariant. No retired pledge/flag remains caller-owned. The exact frame
   survives under the SAME dependency, with no callback to change it. *)
noextract
fn release
  (#a:Type0)
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (value:tensor a)
  (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** runtime stream **
    Epoch.epoch_live stream epoch
  requires owner stream value **
    Pledge.pledge0 (Epoch.epoch_done stream epoch) (contents value ** frame)
  ensures Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
