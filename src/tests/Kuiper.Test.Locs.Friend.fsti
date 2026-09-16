module Kuiper.Test.Locs.Friend

#lang-pulse

(* A friend module needs an interface; none of its implementation details
   should escape through the regression test. *)
inline_for_extraction let () = ()
