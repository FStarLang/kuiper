module Kuiper.PtsTo

include Pulse.Class.PtsTo
#lang-pulse

noeq
type cell a i =
  | Cell : a -> i -> cell a i

// [@@pulse_unfold; noinst]
// instance cell_pts_to (p a : Type) (d : has_pts_to p a) : has_pts_to p (frac a) = {
//   pts_to = (fun p #f' (Frac f v) -> d.pts_to p #(f' *. f) v);
// }
