module Kuiper.Frame
#lang-pulse
open Pulse

(* random utils *)

inline_for_extraction noextract
fn frame_right1 (fr1 : slprop) (#p #q :slprop)
  (f : unit -> stt unit p (fun _ -> q))
  requires p ** fr1
  ensures  q ** fr1
  { f (); }

inline_for_extraction noextract
fn frame_right2 (fr1 fr2 : slprop) (#p #q :slprop)
  (f : unit -> stt unit p (fun _ -> q))
  requires p ** fr1 ** fr2
  ensures  q ** fr1 ** fr2
  { f (); }

inline_for_extraction noextract
fn frame_right3 (fr1 fr2 fr3 : slprop) (#p #q :slprop)
  (f : unit -> stt unit p (fun _ -> q))
  requires p ** fr1 ** fr2 ** fr3
  ensures  q ** fr1 ** fr2 ** fr3
  { f (); }

inline_for_extraction noextract
fn frame_right4 (fr1 fr2 fr3 fr4 : slprop) (#p #q :slprop)
  (f : unit -> stt unit p (fun _ -> q))
  requires p ** fr1 ** fr2 ** fr3 ** fr4
  ensures  q ** fr1 ** fr2 ** fr3 ** fr4
  { f (); }

ghost fn emp_elim  (#f:slprop) ()
  requires emp ** f
  ensures  f
  { () }
ghost fn emp_intro (#f:slprop) ()
  requires f
  ensures  emp ** f
  { () }
ghost fn emp_elim_r  (#f:slprop) ()
  requires f ** emp
  ensures  f
  { () }
ghost fn emp_intro_r (#f:slprop) ()
  requires f
  ensures  f ** emp
  { () }
