module Kuiper.Example.LaunchBounds

#lang-pulse
open Kuiper

(* Extraction regression: each constant launch has its own bound, while a
   runtime-sized launch and a zero-sized launch have no launch-bound attribute. *)

inline_for_extraction noextract
fn empty_kernel (nthr : sz) (tid : szlt nthr) ()
  norewrite
  preserves gpu ** emp ** thread_id nthr tid
{ () }

ghost
fn block_setup (n : erased nat) ()
  norewrite
  requires emp
  ensures (forall+ (tid : natlt n). emp) ** emp
{
  forevery_emp_intro (natlt n);
}

ghost
fn block_teardown (n : erased nat) ()
  norewrite
  requires (forall+ (tid : natlt n). emp) ** emp
  ensures emp
{
  forevery_emp_elim (natlt n);
}

inline_for_extraction noextract
let kdesc (nthr : sz { nthr <= max_threads }) : kernel_desc_1_n emp emp =
{
  nthr = nthr;
  kpre = (fun _ -> emp);
  kpost = (fun _ -> emp);
  f = empty_kernel nthr;
  frame = emp;
  block_setup = block_setup (hide (FStar.SizeT.v nthr));
  block_teardown = block_teardown (hide (FStar.SizeT.v nthr));
  kpost_sendable = solve;
  kpre_sendable = solve;
  full_post_sendable = solve;
  full_pre_sendable = solve;
}

inline_for_extraction noextract
fn launch (nthr : sz { nthr <= max_threads })
  preserves cpu
{
  launch_sync (kdesc nthr);
}

fn fixed32 () preserves cpu { launch 32sz; }
fn fixed64 () preserves cpu { launch 64sz; }
fn dynamic (nthr : sz { nthr <= max_threads }) preserves cpu { launch nthr; }
fn zero () preserves cpu { launch 0sz; }
