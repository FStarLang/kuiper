module Kuiper.Tensor.Runtime.Helpers
#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Send
module RT = Kuiper.Tensor.Runtime
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module KL = Kuiper.Locs
module Pledge = Pulse.Lib.Pledge

inline_for_extraction noextract
instance send_contents (#a:Type0) (value:RT.tensor a)
  : is_send (RT.contents value) = FStar.Tactics.Typeclasses.solve

inline_for_extraction noextract
let rec send_all_contents_aux (#a:Type0) (values:list (RT.tensor a))
  : Tot (is_send (all_contents values)) (decreases values) =
  match values with
  | [] -> is_send_placeless emp
  | value::rest ->
    is_send_star (RT.contents value) (all_contents rest)
      #(send_contents value) #(send_all_contents_aux rest)

inline_for_extraction noextract
instance send_all_contents (#a:Type0) (values:list (RT.tensor a))
  : is_send (all_contents values) = send_all_contents_aux values

inline_for_extraction noextract
ghost
fn uncons_contents
  (#a:Type0) (value:RT.tensor a) (rest:list (RT.tensor a))
  (frame:slprop) ()
  requires all_contents (value::rest) ** frame
  ensures RT.contents value ** (all_contents rest ** frame)
{
  unfold all_contents (value::rest);
}

inline_for_extraction noextract
ghost
fn nil_contents (#a:Type0) (frame:slprop) ()
  requires all_contents #a [] ** frame
  ensures frame
{
  unfold all_contents #a [];
}

inline_for_extraction noextract
fn rec release_ordered
  (#a:Type0) (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (values:list (RT.tensor a)) (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires all_owners stream values **
    Pledge.pledge0 (Epoch.epoch_done stream epoch) (all_contents values ** frame)
  ensures Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
  decreases values
{
  match values {
    [] -> {
      unfold all_owners stream [];
      Pledge.rewrite_pledge (all_contents #a [] ** frame) frame
        (nil_contents #a frame);
    }
    value::rest -> {
      unfold all_owners stream (value::rest);
      Pledge.rewrite_pledge (all_contents (value::rest) ** frame)
        (RT.contents value ** (all_contents rest ** frame))
        (uncons_contents value rest frame);
      RT.release stream #epoch value (all_contents rest ** frame) ();
      release_ordered stream #epoch rest frame ();
    }
  }
}
