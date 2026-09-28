module Kuiper.Externs
#lang-pulse
open Pulse.Lib.Pervasives
open Kuiper

(** The C and CUDA vocabulary Kuiper's extraction rules emit.

    Nothing in Kuiper calls anything declared here, and nothing should: these
    are not an API.  They exist so that the extraction plugin has something to
    point at.

    A Custard rule (see [extraction/KuiperCustard.fst]) is consulted in step 1
    of the extraction loop, before a name's definition is looked up, and builds
    a target-language term out of the call's arguments.  To emit a call to,
    say, [KPR_KCALL], the rule constructs an [EQual] naming a *specific* F*
    lid -- so every C symbol a rule can emit needs a declaration somewhere for
    that lid to resolve to.  This file is that set, in one place, each carrying
    the [custard_extern] target spelling and the [custard_c_header] that makes
    the right [#include] appear.

    Two consequences worth knowing:

    - Because no source call reaches them, they would all be dropped by
      dead-code elimination.  [KuiperCustard.fst] therefore [register_root]s
      each one; forgetting to is error 379.

    - They are declared in [FStar.All.ML], not [Tot], and that is load-bearing
      twice over.  A pure extern returning [unit] is dead code and Custard
      deletes the call outright, and a pure extern returning a value is
      [reeval]-able, so duplicate calls are shared -- fatal for
      [KPR_MEMCPY_H2D] or [KPR_GPU_ALLOC], whose whole point is the effect.
      [Tot] would also make them functions in the *logic*, where
      [kpr_gpu_alloc n == kpr_gpu_alloc n] becomes provable: two distinct
      allocations equal, which is a contradiction waiting to be used.  [ML]
      keeps them out of the logic entirely. *)

(* [KPR_KCALL] is a variadic C macro.  A Custard [external] has a fixed arity,
   so there is one declaration per capture count -- but it need not be fixed in
   its *types*: an external may be polymorphic as long as [custard_c_header]
   names the header, in which case Custard emits no prototype of its own and
   every type vector shares the one macro (error 384 spells this out).  So one
   declaration per arity covers every kernel, instead of one per type vector. *)
[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall (k : unit -> FStar.All.ML unit)
                  (nblk nthr smem : Kuiper.SizeT.t)
                  (s : Kuiper.Kernel.Stream.stream_t) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall1 (#a1 : Type0)
                  (k : unit -> FStar.All.ML unit)
                  (nblk nthr smem : Kuiper.SizeT.t)
                  (s : Kuiper.Kernel.Stream.stream_t)
                  (x1 : a1) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall2 (#a1 #a2 : Type0)
                  (k : unit -> FStar.All.ML unit)
                  (nblk nthr smem : Kuiper.SizeT.t)
                  (s : Kuiper.Kernel.Stream.stream_t)
                  (x1 : a1)
                  (x2 : a2) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall3 (#a1 #a2 #a3 : Type0)
                  (k : unit -> FStar.All.ML unit)
                  (nblk nthr smem : Kuiper.SizeT.t)
                  (s : Kuiper.Kernel.Stream.stream_t)
                  (x1 : a1)
                  (x2 : a2)
                  (x3 : a3) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall4 (#a1 #a2 #a3 #a4 : Type0)
                  (k : unit -> FStar.All.ML unit)
                  (nblk nthr smem : Kuiper.SizeT.t)
                  (s : Kuiper.Kernel.Stream.stream_t)
                  (x1 : a1)
                  (x2 : a2)
                  (x3 : a3)
                  (x4 : a4) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall5 (#a1 #a2 #a3 #a4 #a5 : Type0)
                  (k : unit -> FStar.All.ML unit)
                  (nblk nthr smem : Kuiper.SizeT.t)
                  (s : Kuiper.Kernel.Stream.stream_t)
                  (x1 : a1)
                  (x2 : a2)
                  (x3 : a3)
                  (x4 : a4)
                  (x5 : a5) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall6 (#a1 #a2 #a3 #a4 #a5 #a6 : Type0)
                  (k : unit -> FStar.All.ML unit)
                  (nblk nthr smem : Kuiper.SizeT.t)
                  (s : Kuiper.Kernel.Stream.stream_t)
                  (x1 : a1)
                  (x2 : a2)
                  (x3 : a3)
                  (x4 : a4)
                  (x5 : a5)
                  (x6 : a6) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall7 (#a1 #a2 #a3 #a4 #a5 #a6 #a7 : Type0)
                  (k : unit -> FStar.All.ML unit)
                  (nblk nthr smem : Kuiper.SizeT.t)
                  (s : Kuiper.Kernel.Stream.stream_t)
                  (x1 : a1)
                  (x2 : a2)
                  (x3 : a3)
                  (x4 : a4)
                  (x5 : a5)
                  (x6 : a6)
                  (x7 : a7) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall8 (#a1 #a2 #a3 #a4 #a5 #a6 #a7 #a8 : Type0)
                   (k : unit -> FStar.All.ML unit)
                   (nblk nthr smem : Kuiper.SizeT.t)
                   (s : Kuiper.Kernel.Stream.stream_t)
                   (x1 : a1)
                   (x2 : a2)
                   (x3 : a3)
                   (x4 : a4)
                   (x5 : a5)
                   (x6 : a6)
                   (x7 : a7)
                   (x8 : a8) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall9 (#a1 #a2 #a3 #a4 #a5 #a6 #a7 #a8 #a9 : Type0)
                   (k : unit -> FStar.All.ML unit)
                   (nblk nthr smem : Kuiper.SizeT.t)
                   (s : Kuiper.Kernel.Stream.stream_t)
                   (x1 : a1)
                   (x2 : a2)
                   (x3 : a3)
                   (x4 : a4)
                   (x5 : a5)
                   (x6 : a6)
                   (x7 : a7)
                   (x8 : a8)
                   (x9 : a9) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall10 (#a1 #a2 #a3 #a4 #a5 #a6 #a7 #a8 #a9 #a10 : Type0)
                   (k : unit -> FStar.All.ML unit)
                   (nblk nthr smem : Kuiper.SizeT.t)
                   (s : Kuiper.Kernel.Stream.stream_t)
                   (x1 : a1)
                   (x2 : a2)
                   (x3 : a3)
                   (x4 : a4)
                   (x5 : a5)
                   (x6 : a6)
                   (x7 : a7)
                   (x8 : a8)
                   (x9 : a9)
                   (x10 : a10) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall11 (#a1 #a2 #a3 #a4 #a5 #a6 #a7 #a8 #a9 #a10 #a11 : Type0)
                   (k : unit -> FStar.All.ML unit)
                   (nblk nthr smem : Kuiper.SizeT.t)
                   (s : Kuiper.Kernel.Stream.stream_t)
                   (x1 : a1)
                   (x2 : a2)
                   (x3 : a3)
                   (x4 : a4)
                   (x5 : a5)
                   (x6 : a6)
                   (x7 : a7)
                   (x8 : a8)
                   (x9 : a9)
                   (x10 : a10)
                   (x11 : a11) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_KCALL";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kcall12 (#a1 #a2 #a3 #a4 #a5 #a6 #a7 #a8 #a9 #a10 #a11 #a12 : Type0)
                   (k : unit -> FStar.All.ML unit)
                   (nblk nthr smem : Kuiper.SizeT.t)
                   (s : Kuiper.Kernel.Stream.stream_t)
                   (x1 : a1)
                   (x2 : a2)
                   (x3 : a3)
                   (x4 : a4)
                   (x5 : a5)
                   (x6 : a6)
                   (x7 : a7)
                   (x8 : a8)
                   (x9 : a9)
                   (x10 : a10)
                   (x11 : a11)
                   (x12 : a12) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "blockIdx.x";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_blockidx : Kuiper.SizeT.t

[@@FStar.Attributes.custard_extern "threadIdx.x";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_threadidx : Kuiper.SizeT.t

(* The base of the block's dynamic shared memory, plus a byte offset.  The
   rule casts the result to each request's element type. *)
[@@FStar.Attributes.custard_extern "KPR_SHMEM_AT";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_shmem_at (off : Kuiper.SizeT.t)
  : FStar.All.ML (Pulse.Lib.Array.array FStar.UInt8.t)

(* The GPU allocator, in bytes-per-element and element-count form, exactly as
   the Krml plugin spelled it (ExtractKuiper.fst:977-980).  The rule casts the
   result to the element type. *)
[@@FStar.Attributes.custard_extern "KPR_GPU_ALLOC";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_gpu_alloc (sz : Kuiper.SizeT.t) (len : Kuiper.SizeT.t)
  : FStar.All.ML (Pulse.Lib.Array.array FStar.UInt8.t)

[@@FStar.Attributes.custard_extern "vec_memcpy";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val vec_memcpy (#a #b : Type0)
                      (dst : Pulse.Lib.Array.array a)
                      (src : Pulse.Lib.Array.array b) : FStar.All.ML unit

(* ---- The host-side runtime (section 79) ---------------------------------
   The CUDA runtime calls the Krml plugin emitted directly
   (ExtractKuiper.fst:964-1030).  Each is a macro rather than a function so
   that MUST() can report the failing call site. *)

[@@FStar.Attributes.custard_extern "KPR_MEMCPY_H2D";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_memcpy_h2d (#a #b : Type0)
                          (dst : Pulse.Lib.Array.array a)
                          (src : Pulse.Lib.Array.array b)
                          (bytes : Kuiper.SizeT.t) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_MEMCPY_D2H";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_memcpy_d2h (#a #b : Type0)
                          (dst : Pulse.Lib.Array.array a)
                          (src : Pulse.Lib.Array.array b)
                          (bytes : Kuiper.SizeT.t) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_MEMCPY_D2D";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_memcpy_d2d (#a #b : Type0)
                          (dst : Pulse.Lib.Array.array a)
                          (src : Pulse.Lib.Array.array b)
                          (bytes : Kuiper.SizeT.t) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_GPU_FREE";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_gpu_free (#a : Type0)
                        (p : Pulse.Lib.Array.array a) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_GUARD";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_guard (b : bool) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_ASSERT";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_assert (b : bool) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_SYNC_DEVICE";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_sync_device (u : unit) : FStar.All.ML unit

(* The rest of the launch prologue (ExtractKuiper.fst:450-482).  [KPR_KCALL]
   alone reserves the shared memory but does not check it against the device
   limit, and does not raise the 48KiB default cap -- so a kernel asking for
   more than 48KiB silently fails to launch.  As with [kcall] above, the
   kernel's F* type here is a placeholder: the macro is variadic in C and the
   rule supplies the real one. *)

[@@FStar.Attributes.custard_extern "KPR_SHMEM_FITS";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_shmem_fits (n : Kuiper.SizeT.t) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_SET_MAX_DYN_SHMEM";
   FStar.Attributes.custard_c_header "kuiper.h"]
assume val kpr_set_max_dyn_shmem (k : unit -> FStar.All.ML unit)
                                 (bytes : Kuiper.SizeT.t) : FStar.All.ML unit

(* ---- TensorCore (section 66 experiment) --------------------------------- *)

(* [wmma::fragment<...>] is a C++ *template* type whose arguments are the
   fragment's kind, its m/n/k and its layout.  Those are F* type indices on
   [Kuiper.TensorCore.Base.fragment], and Custard's [cty] for that type keeps
   only the type argument -- so the C type of a fragment local cannot be
   spelled from its Custard type alone.  This is the probe for that: a
   fragment-typed local has to be *declared*, and this is what declares it. *)
(* Hopper WGMMA.  Unlike the wmma entry points these take the fragment by
   reference in C++ and are ordinary functions rather than macros, so the
   [custard_extern] name is the function's own. *)
[@@FStar.Attributes.custard_extern "kpr_wgmma_fill";
   FStar.Attributes.custard_c_header "kuiper/wgmma.h"]
assume val kpr_wgmma_fill (#f : Type0) (fr : f) (x : Kuiper.Float32.t)
  : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "kpr_wgmma_load_accum";
   FStar.Attributes.custard_c_header "kuiper/wgmma.h"]
assume val kpr_wgmma_load_accum (#f #a : Type0) (fr : f)
                                (c : Pulse.Lib.Array.array a)
                                (stride : Kuiper.SizeT.t) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "kpr_wgmma_store";
   FStar.Attributes.custard_c_header "kuiper/wgmma.h"]
assume val kpr_wgmma_store (#f #a : Type0) (fr : f)
                           (c : Pulse.Lib.Array.array a)
                           (stride : Kuiper.SizeT.t) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "kpr_wgmma_mma_sync";
   FStar.Attributes.custard_c_header "kuiper/wgmma.h"]
assume val kpr_wgmma_mma_sync (#f #a : Type0) (a_ b_ : Pulse.Lib.Array.array a)
                              (fr : f) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "wmma::mma_sync";
   FStar.Attributes.custard_c_header "kuiper/tensorcores.h"]
assume val kpr_mma_sync (#a : Type0) (d x y c : a) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_LOAD_ACCUM";
   FStar.Attributes.custard_c_header "kuiper/tensorcores.h"]
assume val kpr_load_accum (#f #a : Type0) (fr : f)
                          (gm : Pulse.Lib.Array.array a)
                          (ldm : Kuiper.SizeT.t) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "wmma::load_matrix_sync";
   FStar.Attributes.custard_c_header "kuiper/tensorcores.h"]
assume val kpr_load_ab (#f #a : Type0) (fr : f)
                       (gm : Pulse.Lib.Array.array a)
                       (ldm : Kuiper.SizeT.t) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "KPR_STORE";
   FStar.Attributes.custard_c_header "kuiper/tensorcores.h"]
assume val kpr_store (#f #a : Type0) (gm : Pulse.Lib.Array.array a)
                     (fr : f) (ldm : Kuiper.SizeT.t) : FStar.All.ML unit

[@@FStar.Attributes.custard_extern "wmma::fill_fragment";
   FStar.Attributes.custard_c_header "kuiper/tensorcores.h"]
assume val kpr_fill (#f #a : Type0) (fr : f) (v : a) : FStar.All.ML unit
