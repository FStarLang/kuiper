module Kuiper.PtsTo

open FStar.Tactics.Typeclasses
open Pulse
module R = Kuiper.Ref
module A = Kuiper.Array

(* See FStarLang/FStar#3522, if this class is named 'pts_to', then things break
very oddly. *)

[@@fundeps [1]; pulse_unfold]
class has_pts_to (p r :Type) = {
  [@@@pulse_unfold]
  ( |-> ) : p -> r -> Tot slprop;
}

(* These have to be exposed if we want to use them in specs. *)

[@@pulse_unfold]
unfold
instance has_pts_to_erased (p r : Type) (_ : has_pts_to p r) : has_pts_to p (erased r) = {
  ( |-> ) = (fun p v -> p |-> reveal v);
}


[@@pulse_unfold]
unfold
instance has_pts_to_ref (a:Type) : has_pts_to (ref a) a = {
  ( |-> ) = (fun r v -> Pulse.Lib.Reference.pts_to r v);
}

[@@pulse_unfold]
unfold
instance has_pts_to_gref (a:Type) : has_pts_to (Pulse.Lib.GhostReference.ref a) a = {
  ( |-> ) = (fun r v -> Pulse.Lib.GhostReference.pts_to r v);
}

[@@pulse_unfold]
unfold
instance has_pts_to_arr (a:Type) : has_pts_to (array a) (Seq.seq a) = {
  ( |-> ) = (fun r v -> Pulse.Lib.Array.pts_to r v);
}

[@@pulse_unfold]
unfold
instance has_pts_to_gpu_ref (a:Type) : has_pts_to (R.gpu_ref a) a = {
  ( |-> ) = (fun r v -> R.gpu_pts_to r v);
}

[@@pulse_unfold]
unfold
instance has_pts_to_gpu_arr (a:Type) (sz : _) : has_pts_to (A.gpu_array a sz) (Seq.seq a) = {
  ( |-> ) = (fun r v -> A.gpu_pts_to_array r v);
}
