module Kuiper.Kernel.LogSoftmax

#lang-pulse
open Kuiper
open Kuiper.Real { rlog }
open Kuiper.Seq.Common
module SZ = Kuiper.SizeT
module Vec = Pulse.Lib.Vec
module SM = Kuiper.Kernel.Softmax

// Log of softmax.
let log_softmax_real (s:Seq.seq real { Seq.length s > 0 }) =
  lseq_map rlog (seq_refine (fun x -> x >. 0.0R) (SM.softmax_real s))

unfold
type log_softmax_ty (et : Type0) {| floating et, real_like et |} =
  fn (nth : szp{nth <= max_threads})
     (#lena : szp)
     (a : Vec.lvec et lena)
     (#va : erased (lseq et lena))
     (ra : erased (lseq real lena))
  preserves
    cpu
  requires
    a |-> va **
    pure (va %~ ra) **
    pure (lena <= max_blocks * max_threads)
    // ^ This could be removed
  ensures
    exists* (va' : lseq et lena).
      a |-> va' **
      pure (va' %~ log_softmax_real ra)

inline_for_extraction noextract
val log_softmax (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  : log_softmax_ty et
