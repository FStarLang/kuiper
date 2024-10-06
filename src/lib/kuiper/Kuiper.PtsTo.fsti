module Kuiper.PtsTo

open FStar.Tactics.Typeclasses
open Pulse
module R = Kuiper.Ref
module A = Kuiper.Array

[@@pulse_unfold]
unfold
instance has_pts_to_gref (a:Type) : has_pts_to (Pulse.Lib.GhostReference.ref a) a = {
  pts_to = Pulse.Lib.GhostReference.pts_to;
}

[@@pulse_unfold]
unfold
instance has_pts_to_gpu_ref (a:Type) : has_pts_to (R.gpu_ref a) a = {
  pts_to = R.gpu_pts_to;
}

[@@pulse_unfold]
unfold
instance has_pts_to_gpu_arr (a:Type) (sz : _) : has_pts_to (A.gpu_array a sz) (Seq.seq a) = {
  pts_to = A.gpu_pts_to_array;
}
