module KPlay.Functional
#lang-pulse

include KPlay.Proof.RouteSteps

open Kuiper
open Pulse.Lib.Send

module Program = KPlay.Proof.SourceProgram
module Binding = KPlay.Proof.NativeBindings
module Source = KPlay.Proof.SourceLayouts
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch

inline_for_extraction noextract
fn program_refines
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens:Binding.token_argument) (weights:Binding.weight_argument)
  (#x:erased Source.token_values) (#w:erased Source.weight_values) (#fp #fw:perm)
  (frame:slprop) {| is_send frame |} ()
  requires cpu ** Program.before_call stream epoch tokens weights x w fp fw frame
  returns result:Program.outcome
  ensures cpu ** Program.after_call stream epoch tokens weights x w fp fw frame result
{
  Program.run stream #epoch tokens weights #x #w #fp #fw frame ()
}
