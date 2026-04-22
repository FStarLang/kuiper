module Kuiper.Kernel.LogSoftmax

#lang-pulse
open Kuiper
open Kuiper.Real { rlog }
open Kuiper.Seq.Common
module Vec = Pulse.Lib.Vec
module SM = Kuiper.Kernel.Softmax

// Log of softmax.
let log_softmax_real (s:Seq.seq real { Seq.length s > 0 }) =
  lseq_map rlog (seq_refine (fun x -> x >. 0.0R) (SM.softmax_real s))

unfold
type log_softmax_ty (et : Type0) {| floating et, real_like et |} =
  fn (#lena : szp)
     (a : Vec.lvec et lena)
     (#va : erased (lseq et lena))
     (ra : erased (lseq real lena))
  preserves
    cpu
  requires
    a |-> va **
    pure (lena <= max_threads) **
    pure (va %~ ra)
  ensures
    exists* (va' : lseq et lena).
      a |-> va' **
      pure (va' %~ log_softmax_real ra)

inline_for_extraction noextract
val log_softmax (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  : log_softmax_ty et
