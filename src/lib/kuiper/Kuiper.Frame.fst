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

inline_for_extraction noextract
fn frame_2left (fr : slprop) (#p1 #p2 #q1 #q2 :slprop)
  (f : unit -> stt unit (p1 ** p2) (fun _ -> q1 ** q2))
  requires p1 ** p2 ** fr
  ensures  q1 ** q2 ** fr
  { f (); }

inline_for_extraction noextract
fn frame_3left (fr : slprop) (#p1 #p2 #p3 #q1 #q2 #q3 :slprop)
  (f : unit -> stt unit (p1 ** p2 ** p3) (fun _ -> q1 ** q2 ** q3))
  requires p1 ** p2 ** p3 ** fr
  ensures  q1 ** q2 ** q3 ** fr
  { f (); }

ghost fn emp_elim_l  (#f:slprop) ()
  requires emp ** f
  ensures  f
  { () }
ghost fn emp_intro_l (#f:slprop) ()
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

ghost fn emp_elim_r2  (#f1 #f2 : slprop) ()
  requires f1 ** f2 ** emp
  ensures  f1 ** f2
  { () }
ghost fn emp_intro_r2 (#f1 #f2 : slprop) ()
  requires f1 ** f2
  ensures  f1 ** f2 ** emp
  { () }
