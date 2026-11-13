module Kuiper.Tensor.Input
#lang-pulse

open Pulse.Lib.Core
module A = Pulse.Lib.Array
module KA = Kuiper.Array.Core
module KL = Kuiper.Locs
module RT = Kuiper.Tensor.Runtime
module I64 = FStar.Int64

inline_for_extraction noextract
let _ = ()

(* EXTERNAL NATIVE INPUT/ABI TRUST. An argument denotes the actual source
   Tensor reference and its fixed, positive, contiguous cuda:0 view. It is
   distinct from Runtime.tensor, which names an allocated storage generation.
   The source entry binds these names and reference permissions to its live
   Tensor arguments; GPU cell ownership alone cannot manufacture that binding.

   The binding retains TensorImpl -> StorageImpl -> DataPtr for the call, even
   when the view has a nonzero storage offset or its parent Tensor handle has
   gone away. Header mutation, pointer materializers, access-error/warning
   hooks and untracked alias actions are outside this direct-view profile.
   In particular the actual mutable data_ptr<T> is not silently replaced by
   const_data_ptr<T>: a materializing call is NOT claimed to be a pure query.

   reference grants neither GPU cells nor final-reference release authority.
   There is no constructor/import from arrays, no reference-to-owner conversion,
   no raw-free permission, and no final-owner destructor in this borrowed API.
   with_parameter models the existing native by-value parameter scope, not a
   new CUDA operation or a constructor from mathematical input-cell ownership. *)
val argument (a:Type0) : Type0
val kind (#a:Type0) (value:argument a) : GTot (RT.dtype a)
val dimensions (#a:Type0) (value:argument a)
  : GTot (shape:list pos{RT.numel shape <= 2147483647})
val backing (#a:Type0) (value:argument a)
  : GTot (array:A.array a{KA.is_global_array array})
val offset (#a:Type0) (value:argument a)
  : GTot (start:nat{start + RT.numel (dimensions value) <= A.length (backing value)})
val reference (#a:Type0) (value:argument a) : slprop

(* EXTERNAL SOURCE-ABI TRUST: value is the actual native Tensor value at the
   by-value call boundary. Its strong parameter reference retains the backing
   throughout body; the balanced reference cannot escape in post. This is a
   bracket for the existing C++ parameter lifetime, not an inserted wrapper,
   runtime guard, allocation, or assumption that GPU work has completed.
   The caller's GPU permissions/epoch loans remain separate obligations. *)
noextract
fn with_parameter
  (#a #r:Type0) (value:argument a)
  (pre:slprop) (post:r -> slprop)
  (body: fn ()
    requires KL.cpu ** reference value ** pre
    returns result:r
    ensures KL.cpu ** reference value ** post result)
  requires KL.cpu ** pre
  returns result:r
  ensures KL.cpu ** post result

inline_for_extraction noextract
let data (#a:Type0) (value:argument a) : GTot (A.array a) =
  A.gsub (backing value) (offset value) (offset value + RT.numel (dimensions value))

inline_for_extraction noextract
let view_metadata (#a:Type0) (value:argument a)
  : Lemma (
      A.length (data value) == RT.numel (dimensions value) /\
      KA.is_global_array (data value) /\
      A.base_of (data value) == A.base_of (backing value) /\
      A.offset_of (data value) == A.offset_of (backing value) + offset value) =
  A.length_gsub (backing value) (offset value) (offset value + RT.numel (dimensions value));
  A.loc_id_of_array_gsub (backing value) (offset value) (offset value + RT.numel (dimensions value));
  A.visibility_of_array_gsub (backing value) (offset value) (offset value + RT.numel (dimensions value));
  A.base_of_gsub (backing value) (offset value) (offset value + RT.numel (dimensions value));
  A.offset_of_gsub (backing value) (offset value) (offset value + RT.numel (dimensions value))

(* Actual Tensor::data_ptr<T>() on the source argument. Under the direct-view
   profile it neither accesses GPU cells nor enqueues work or synchronizes.
   Type/shape/liveness are native-ABI validity, not an allocation-success or
   kernel-correctness assumption. Unsupported native errors are not converted
   into successful pointer results by an exception-catching wrapper. *)
noextract
fn data_ptr
  (#a:Type0) (value:argument a)
  preserves KL.cpu ** reference value
  returns address:A.array a
  ensures pure (address == data value)

(* The source input1.numel() is int64, before the caller's checked narrowing. *)
noextract
fn numel
  (#a:Type0) (value:argument a)
  preserves KL.cpu ** reference value
  returns count:I64.t
  ensures pure (I64.v count == RT.numel (dimensions value))
