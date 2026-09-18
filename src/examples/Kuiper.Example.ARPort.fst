module Kuiper.Example.ARPort
#lang-pulse
open Pulse.Lib.Pervasives
open Kuiper

(* The runtime entries the Custard rule's emitted code names.  Nothing in the
   source calls them, so the rule must [register_root] each one. *)

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

let reverse_u64 = Kuiper.Example.ArrayReversal.reverse #FStar.UInt64.t

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
