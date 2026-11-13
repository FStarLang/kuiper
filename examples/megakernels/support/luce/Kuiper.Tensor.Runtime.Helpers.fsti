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
let rec all_contents (#a:Type0) (values:list (RT.tensor a)) : Tot slprop =
  match values with
  | [] -> emp
  | value::rest -> RT.contents value ** all_contents rest

inline_for_extraction noextract
let rec all_owners (#a:Type0) (stream:Stream.stream_t)
  (values:list (RT.tensor a)) : Tot slprop =
  match values with
  | [] -> emp
  | value::rest -> RT.owner stream value ** all_owners stream rest

inline_for_extraction noextract
instance val send_contents (#a:Type0) (value:RT.tensor a)
  : is_send (RT.contents value)

inline_for_extraction noextract
instance val send_all_contents (#a:Type0) (values:list (RT.tensor a))
  : is_send (all_contents values)

(* List order is release order. The unchanged frame may contain precise input,
   output and history witnesses; none is interpreted by this adapter. *)
inline_for_extraction noextract
fn release_ordered
  (#a:Type0) (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (values:list (RT.tensor a)) (frame:slprop) {| is_send frame |} ()
  preserves KL.cpu ** Stream.stream_live stream ** RT.runtime stream **
    Epoch.epoch_live stream epoch
  requires all_owners stream values **
    Pledge.pledge0 (Epoch.epoch_done stream epoch) (all_contents values ** frame)
  ensures Pledge.pledge0 (Epoch.epoch_done stream epoch) frame
