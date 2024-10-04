module Kuiper.PtsTo

open FStar.Tactics.Typeclasses
open Pulse
open Kuiper.Array

[@@fundeps [1]; pulse_unfold]
class pts_to (p r :Type) = {
  ( |-> ) : p -> r -> Tot slprop;
}

(* These have to be exposed if we want to use them in specs. *)

[@@pulse_unfold]
unfold
instance has_pts_to_ref (a:Type) : pts_to (ref a) a = {
  ( |-> ) = (fun r v -> Pulse.Lib.Reference.pts_to r v);
}

[@@pulse_unfold]
unfold
instance has_pts_to_arr (a:Type) : pts_to (array a) (Seq.seq a) = {
  ( |-> ) = (fun r v -> Pulse.Lib.Array.pts_to r v);
}

[@@pulse_unfold]
unfold
instance has_pts_to_arr_e (a:Type) : pts_to (array a) (Ghost.erased (Seq.seq a)) = {
  ( |-> ) = (fun r v -> Pulse.Lib.Array.pts_to r (Ghost.reveal v));
}

[@@pulse_unfold]
unfold
instance has_pts_to_gpu_arr (a:Type) (sz : _) : pts_to (gpu_array a sz) (Seq.seq a) = {
  ( |-> ) = (fun r v -> Kuiper.gpu_pts_to_array r v);
}
